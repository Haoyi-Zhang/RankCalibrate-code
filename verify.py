#!/usr/bin/env python3
"""Reproduce all exact finite checks and data tables, without network access.

Generated distributions are analytic validation inputs, not a retrieval benchmark.
Every scientific equality uses Fraction, with no tolerance or random sampling.
"""
import argparse
import csv
import json
import resource
import time
import unittest
from fractions import Fraction as F
from itertools import permutations
from math import comb
from pathlib import Path
from ranking import *
from reference_audit import audit_references

HERE = Path(__file__).resolve().parent


def csv_write(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def expectation(p, values):
    return sum((a*b for a,b in zip(p,values)),F(0))


def exhaustive(out):
    rows=[]
    coefficient_rows=[]
    mutation_mismatches = 0
    for n in range(2,6):
        for b in range(1,n+1):
            acts=list(ordered_partitions(tuple(range(n)),b))
            matrix=[];maxdegree=0;refinement_count=0;max_coefficient_norm=F(0)
            for action in acts:
                validate_action(action,n)
                vals=[]
                for y in range(1,1<<n):
                    # Max over every refinement is definitionally independent
                    # of the sorted-completion and block-polynomial routines.
                    val=oracle_utility(action,y)
                    assert val==sorted_utility(action,y)==numerator(action,y)/y.bit_count()
                    vals.append(val)
                refinement_count += ((1<<n)-1)*sum(1 for _ in refinements(action))
                c=mobius_coefficients([F(0)]+[v*y.bit_count() for y,v in enumerate(vals,1)],n)
                assert c==newton_coefficients(action,n)
                assert c[0]==0 and sum(c)==n
                coefficient_norm=sum(map(abs,c),F(0))
                coefficient_bound=F(n) if b==1 else n*(1+F(2*(2**b-1),b))
                assert coefficient_norm<=coefficient_bound
                max_coefficient_norm=max(max_coefficient_norm,coefficient_norm)
                degree=max(t.bit_count() for t,x in enumerate(c) if x)
                maxdegree=max(maxdegree,degree)
                assert degree<=min(n,b+1)
                if degree>=3:
                    # Negative control: deleting higher interactions must fail.
                    for y,val in enumerate(vals,1):
                        truncated=sum((c[t] for t in range(1,1<<n) if t.bit_count()<=2 and y&t==t),F(0))/y.bit_count()
                        mutation_mismatches += int(truncated != val)
                matrix.append(vals)
            contrasts=[[x-z for x,z in zip(row,matrix[0])] for row in matrix[1:]]
            rank,bits=exact_rank(contrasts)
            assert rank==dimension(n,b)
            basis=contrast_basis(n,b)
            assert len(basis)==rank
            features=[contrast_features(n,b,y) for y in range(1,1<<n)]
            for action,vals in zip(acts,matrix):
                coef=contrast_coefficients(action,acts[0],n,b)
                for y,row in enumerate(features,1):
                    assert expectation(coef,row)==vals[y-1]-matrix[0][y-1]
            nu=reference_distribution(n)
            centered=[[x-expectation(nu,row) for x in row] for row in contrasts]
            centered_rank,centered_bits=exact_rank(centered)
            augmented_rank,augmented_bits=exact_rank([[F(1)]*((1<<n)-1)]+centered)
            assert centered_rank==rank and augmented_rank==rank+1
            bits=max(bits,centered_bits,augmented_bits)
            p=reference_distribution(n)
            kappa=(n+harmonic(0,n))/(2*n)
            for action,vals in zip(acts,matrix):
                fee=action_fee(action,lambda t,s:reference_fee(n,t,s))
                assert fee>=0
                assert expectation(p,vals)-fee==kappa
                assert fee==expectation(p,[x-average_precision(tuple(i for block in action for i in block),y) for y,x in enumerate(vals,1)])
            fields=['action']+[f'y{y}' for y in range(1,1<<n)]
            csv_write(out/'matrices'/f'n{n}_b{b}.csv',fields,
                [dict(action=action_name(a),**{f'y{y}':str(v) for y,v in enumerate(vals,1)}) for a,vals in zip(acts,matrix)])
            row=dict(n=n,b=b,actions=len(acts),outcomes=(1<<n)-1,
                action_outcome_pairs=len(acts)*((1<<n)-1),refinement_evaluations=refinement_count,
                contrast_rank=rank,predicted_rank=dimension(n,b),max_degree=maxdegree,
                elimination_max_bits=bits,reference_bayes_risk=str(1-kappa))
            coefficient_rows.append(dict(n=n,b=b,max_numerator_coefficient_norm=str(max_coefficient_norm),numerator_bound=str(coefficient_bound),transfer_bound=str(2*coefficient_bound)))
            rows.append(row); print('exhaustive',n,b,'rank',rank,flush=True)
    assert mutation_mismatches>0
    csv_write(out/'dimensions.csv',list(rows[0]),rows)
    csv_write(out/'coefficient_bounds.csv',list(coefficient_rows[0]),coefficient_rows)
    # An outcome-independent uniform completion remains quadratic.
    random_checks=0
    for n in range(2,5):
        for action in ordered_partitions(tuple(range(n)),n):
            orders=list(refinements(action))
            vals=[F(0)]+[sum((average_precision(order,y) for order in orders),F(0))/len(orders)*y.bit_count() for y in range(1,1<<n)]
            coeff=mobius_coefficients(vals,n)
            assert not any(x for t,x in enumerate(coeff) if t.bit_count()>2)
            random_checks+=1
    return dict(cases=len(rows),action_outcome_pairs=sum(r['action_outcome_pairs'] for r in rows),
        refinement_evaluations=sum(r['refinement_evaluations'] for r in rows),
        truncated_quadratic_mismatches=mutation_mismatches,random_completion_actions=random_checks,
        elimination_max_bits=max(r['elimination_max_bits'] for r in rows))


def _modular_independent_rows(rows, target_rank, prime=1_000_000_007):
    """Select a rationally independent row certificate via a prime field."""
    pivots = {}
    selected = []
    for original in rows:
        row = [
            (value.numerator % prime) * pow(value.denominator % prime, prime - 2, prime) % prime
            for value in original
        ]
        for column, pivot in sorted(pivots.items()):
            if row[column]:
                coefficient = row[column]
                row = [(x - coefficient * y) % prime for x, y in zip(row, pivot)]
        column = next((index for index, value in enumerate(row) if value), None)
        if column is None:
            continue
        inverse = pow(row[column], prime - 2, prime)
        row = [(value * inverse) % prime for value in row]
        pivots[column] = row
        selected.append(original)
        if len(selected) == target_rank:
            break
    return selected, len(selected), prime


def extended_geometry(out):
    """n=6 coefficient checks with exact rational independence certificates."""
    n = 6
    rows = []
    definition_samples = 0
    for b in range(1, n + 1):
        actions = list(ordered_partitions(tuple(range(n)), b))
        reference = actions[0]
        reference_coefficients = newton_coefficients(reference, n)
        all_contrasts = []
        maximum_degree = 0
        reference_net_values = set()
        sample_indices = sorted({0, len(actions) // 2, len(actions) - 1})
        for index, action in enumerate(actions):
            coefficients = newton_coefficients(action, n)
            contrast = [
                value - reference_value
                for value, reference_value in zip(coefficients, reference_coefficients)
            ]
            all_contrasts.append(contrast)
            maximum_degree = max(
                maximum_degree,
                max(term.bit_count() for term, value in enumerate(coefficients) if value),
            )
            assert not any(
                value for term, value in enumerate(contrast)
                if term.bit_count() > min(n, b + 1)
            )
            if b == 1:
                assert sum(
                    contrast[term] for term in range(1, 1 << n)
                    if term.bit_count() == 1
                ) == 0
                assert sum(
                    contrast[term] for term in range(1, 1 << n)
                    if term.bit_count() == 2
                ) == 0
            else:
                assert sum(contrast) == 0
            expected_numerator_over_relevance = sum(
                (
                    coefficients[term]
                    * F(1, n * 2 ** (term.bit_count() - 1))
                    for term in range(1, 1 << n)
                    if coefficients[term]
                ),
                F(0),
            )
            reference_net_values.add(
                expected_numerator_over_relevance
                - action_fee(action, lambda t, s: reference_fee(n, t, s))
            )
            if index in sample_indices:
                for y in (1, 21, 42, 63):
                    assert oracle_utility(action, y) == sorted_utility(action, y)
                    assert sorted_utility(action, y) == numerator(action, y) / y.bit_count()
                    definition_samples += 1
        predicted = dimension(n, b)
        selected, modular_rank, prime = _modular_independent_rows(
            all_contrasts, predicted
        )
        assert modular_rank == predicted
        exact_certificate_rank, bits = exact_rank(selected)
        assert exact_certificate_rank == predicted
        assert maximum_degree <= min(n, b + 1)
        assert len(reference_net_values) == 1
        rows.append(dict(
            n=n,
            b=b,
            actions=len(actions),
            exact_certificate_rows=len(selected),
            exact_certificate_rank=exact_certificate_rank,
            predicted_rank=predicted,
            modular_prime=prime,
            max_degree=maximum_degree,
            elimination_max_bits=bits,
            definition_samples=12,
            reference_net_value=str(next(iter(reference_net_values))),
        ))
        print('extended geometry', n, b, 'rank', exact_certificate_rank, flush=True)
    csv_write(out / 'extended_geometry.csv', list(rows[0]), rows)
    return dict(
        cases=len(rows),
        max_n=n,
        largest_menu=max(row['actions'] for row in rows),
        exact_certificate_rows=sum(row['exact_certificate_rows'] for row in rows),
        definition_samples=definition_samples,
        elimination_max_bits=max(row['elimination_max_bits'] for row in rows),
    )


def fiber(out):
    fixture=json.loads((HERE/'fixtures/fiber.json').read_text())
    p0=[F(x,fixture['base_denominator']) for x in fixture['base_numerators']]
    v=list(map(F,fixture['direction']))
    lo,hi=map(F,fixture['parameter_interval']); lam=F(fixture['pair_fee'])
    def law(t):return [x+t*z for x,z in zip(p0,v)]
    acts=list(ordered_partitions((0,1,2),3))
    rows=[]
    for action in acts:
        vals=[sorted_utility(action,y) for y in range(1,8)]
        intercept=1-expectation(p0,vals)
        slope=-expectation(v,vals)
        charge=sum(comb(len(block),2) for block in action)
        rows.append(dict(action=action_name(action),intercept=str(intercept),t_slope=str(slope),lambda_slope=charge))
    csv_write(out/'fiber_actions.csv',list(rows[0]),rows)
    base_mu=moments(p0,3,2)
    # This rank certificate shows the specified line is the ENTIRE mean fiber.
    features=[[F(1) for y in range(1,8)]]+[[F(int(y&t==t),y.bit_count()) for y in range(1,8)] for t in range(1,7)]
    r,_=exact_rank(features); assert r==6
    for t in (lo, -F(1,120), F(0), F(1,120),hi):
        p=law(t); assert min(p)>=0 and sum(p)==1 and moments(p,3,2)==base_mu
    # Exact vertex tests prove all linear risk comparisons over the rectangle.
    for t in (lo,hi):
        for price in (F(3,80),F(1,20)):
            risks=[F(r['intercept'])+t*F(r['t_slope'])+price*r['lambda_slope'] for r in rows]
            full=F(9,80); deferred=F(17,240)+t/6+price
            assert min(risks)==min(full,deferred)
            # The per-action affine dominance test below supplies the
            # certificate valid throughout the rectangle, not just its corners.
    dominance=[]
    corners=[(t,price) for t in (lo,hi) for price in (F(3,80),F(1,20))]
    for row in rows:
        candidates=[]
        for target in ('full','deferred'):
            gaps=[]
            for t,price in corners:
                risk=F(row['intercept'])+t*F(row['t_slope'])+price*row['lambda_slope']
                bound=F(9,80) if target=='full' else F(17,240)+t/6+price
                gaps.append(risk-bound)
            if min(gaps)>=0:
                candidates.append(dict(target=target,corner_gaps=list(map(str,gaps))))
        assert candidates, (row, 'No affine dominance certificate')
        dominance.append(dict(action=row['action'],certificates=candidates))
    (out/'fiber_dominance.json').write_text(json.dumps(dominance,indent=2)+'\n')
    data=[]
    for k in range(31):
        t=lo+(hi-lo)*F(k,30)
        full=F(9,80); deferred=F(17,240)+t/6+lam
        data.append(dict(t=str(t),t_float=float(t),full=str(full),full_float=float(full),
            deferred=str(deferred),deferred_float=float(deferred),bayes=str(min(full,deferred)),bayes_float=float(min(full,deferred))))
    csv_write(out/'fiber_curve.csv',list(data[0]),data)
    price_rows=[]
    for k in range(21):
        price=F(3,80)+F(k,1600)
        alpha=F(1,20)-price;beta=price-F(3,80)
        assert alpha+beta==F(1,80)
        rand=alpha*beta/(alpha+beta)
        rho=alpha/(alpha+beta)
        assert rho*beta==(1-rho)*alpha==rand
        price_rows.append(dict(pair_fee=str(price),pair_fee_float=float(price),
            deterministic=str(min(alpha,beta)),deterministic_float=float(min(alpha,beta)),
            randomized=str(rand),randomized_float=float(rand),defer_probability=str(rho)))
    csv_write(out/'fee_deficiency.csv',list(price_rows[0]),price_rows)
    # p+ and p- have no common minimizer, even with the all-in-one action.
    witness=[];sets=[]
    for sign in (-1,1):
        t=F(sign,120);p=law(t)
        risks=[F(r['intercept'])+t*F(r['t_slope'])+lam*r['lambda_slope'] for r in rows]
        best=min(risks);opt=[action_name(a) for a,r in zip(acts,risks) if r==best]
        sets.append(set(opt));witness.append(dict(parameter=str(t),probabilities=list(map(str,p)),bayes_risk=str(best),optimal_actions=opt))
    assert not sets[0]&sets[1]
    # Wrong reference pair prices erase the all-tie certificate: fail as intended.
    q=reference_distribution(3)
    wrong=[]
    for action in ordered_partitions((0,1,2),2):
        wrong.append(1-expectation(q,[sorted_utility(action,y) for y in range(1,8)])+
                     action_fee(action,lambda t,s:reference_fee(3,0,s)))
    assert len(set(wrong))>1
    result=dict(fiber_rank=r,parameter_interval=[str(lo),str(hi)],moments={str(t):str(x) for t,x in base_mu.items()},
        pair_fee=str(lam),witnesses=witness,deterministic_deficiency='1/240',randomized_deficiency='1/360',
        optimal_defer_probability='2/3',wrong_prefix_fee_negative_control='detected')
    (out/'fiber_certificate.json').write_text(json.dumps(result,indent=2)+'\n')
    return dict(actions=len(acts),fiber_rank=6,witness_laws=2,price_grid=21,parameter_grid=31)


def parity(out):
    rows=[]
    menu_rows=[]
    for b in range(2,9):
        n=b+1;Z=n*2**(n-2)
        plus=[F(y.bit_count(),Z) if (n-y.bit_count())%2==0 else F(0) for y in range(1,1<<n)]
        minus=[F(y.bit_count(),Z) if (n-y.bit_count())%2 else F(0) for y in range(1,1<<n)]
        assert sum(plus)==sum(minus)==1
        assert moments(plus,n,b)==moments(minus,n,b)
        block=((0,),tuple(range(1,n)));order=tuple(range(n))
        gains=[sorted_utility(block,y)-average_precision(order,y) for y in range(1,1<<n)]
        gp=expectation(plus,gains);gm=expectation(minus,gains)
        gap=F(1,b*(b+1)**2*2**(b-1))
        assert abs(gp-gm)==gap
        top=newton_coefficients(block,n)[-1]
        assert gp-gm==top/Z
        # At the fixed reference-value fees the FULL menu has exactly two
        # conditional-risk levels on this pair of parity distributions.
        # Enumerate the whole menu only through n=6; larger cases above
        # check moments and the named witness, not an all-action enumeration.
        if n<=6:
            a_count=high_count=0
            mu_plus=moments(plus,n,n);mu_minus=moments(minus,n,n)
            kappa=(n+harmonic(0,n))/(2*n)
            for action in ordered_partitions(tuple(range(n)),b):
                c=newton_coefficients(action,n)
                a_count+=1
                assert c[-1] in (F(0),top)
                is_high=(len(action)==2 and len(action[0])==1 and len(action[1])==b)
                assert bool(c[-1])==is_high
                high_count+=is_high
                price=action_fee(action,lambda t,s:reference_fee(n,t,s))
                net_plus=sum((c[t]*mu_plus[t] for t in mu_plus),F(0))-price
                net_minus=sum((c[t]*mu_minus[t] for t in mu_minus),F(0))-price
                assert net_plus==kappa+c[-1]/(2*Z)
                assert net_minus==kappa-c[-1]/(2*Z)
            assert high_count==n
            menu_rows.append(dict(b=b,n=n,actions=a_count,nonzero_top_coefficient_actions=high_count,
                                 deterministic_two_law_floor=str(gap/2),randomized_two_law_floor=str(gap/4)))
        rows.append(dict(b=b,n=n,outcomes=(1<<n)-1,matched_moments=(1<<n)-2,
                         plus_gain=str(gp),minus_gain=str(gm),gap=str(gap),two_action_price=str((gp+gm)/2),top_coefficient=str(top),
                         reference_fee_deterministic_two_law_floor=str(gap/2),reference_fee_randomized_two_law_floor=str(gap/4)))
    csv_write(out/'parity.csv',list(rows[0]),rows)
    csv_write(out/'parity_menus.csv',list(menu_rows[0]),menu_rows)
    return dict(cases=len(rows),largest_n=9,largest_outcome_count=511,full_menu_cases=len(menu_rows),full_menu_max_n=6)


def decoding(out):
    rows=[]
    def distributions(n):
        N=(1<<n)-1
        weights=[((17*y)^(y>>1))%23+1 for y in range(1,N+1)]
        return [('uniform',[F(1,N)]*N),('reference',reference_distribution(n)),
                ('arithmetic',[F(x,sum(weights)) for x in weights])]
    comparisons=0
    for n in range(2,6):
        for b in range(1,n+1):
            acts=list(ordered_partitions(tuple(range(n)),b))
            matrix=[[sorted_utility(a,y) for y in range(1,1<<n)] for a in acts]
            for name,p in distributions(n):
                mu=moments(p,n,min(n,b+1))
                for fee_name,fee in [('zero',lambda t,s:F(0)),('pair',lambda t,s:F(comb(s,2),24)),
                                     ('high',lambda t,s:F(comb(s,2),8)),('reference',lambda t,s:reference_fee(n,t,s))]:
                    best,action,transitions=decode(n,b,mu,fee)
                    enum=min(1-expectation(p,vals)+action_fee(a,fee) for a,vals in zip(acts,matrix))
                    assert best==enum
                    assert best==1-expectation(p,[sorted_utility(action,y) for y in range(1,1<<n)])+action_fee(action,fee)
                    assert transitions==sum(comb(n,s)*2**(n-s) for s in range(1,b+1))
                    comparisons+=1
                    rows.append(dict(n=n,b=b,distribution=name,fee=fee_name,loss=str(best),action=action_name(action),transitions=transitions,verification='all-actions'))
    # Larger cases are DP checks, NOT exhaustive all-action optimum checks.
    for n in (6,7,8):
        b=2;p=distributions(n)[-1][1];mu=moments(p,n,3)
        for fee_name,fee in [('pair',lambda t,s:F(comb(s,2),24)),('reference',lambda t,s:reference_fee(n,t,s))]:
            best,action,transitions=decode(n,b,mu,fee)
            assert best==1-expectation(p,[sorted_utility(action,y) for y in range(1,1<<n)])+action_fee(action,fee)
            assert transitions==sum(comb(n,s)*2**(n-s) for s in range(1,b+1))
            rows.append(dict(n=n,b=b,distribution='arithmetic',fee=fee_name,loss=str(best),action=action_name(action),transitions=transitions,verification='returned-action-and-transition-count'))
    csv_write(out/'decoding.csv',list(rows[0]),rows)
    return dict(exhaustive_optimum_comparisons=comparisons,larger_returned_action_checks=6,max_n=8)


def main():
    if not __debug__:
        raise SystemExit("Do not use Python -O: scientific assertions must remain enabled.")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'results')
    parser.add_argument('--part',choices=['all','exhaustive','extended','fiber','parity','decoding'],default='all')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    suite=unittest.defaultTestLoader.discover(str(HERE/'tests'))
    testresult=unittest.TextTestRunner(verbosity=2).run(suite)
    if not testresult.wasSuccessful():raise SystemExit(1)
    reference_summary = audit_references(
        HERE / 'bibliography.bib', HERE / 'reference_audit.csv',
        HERE / 'external_resources.csv'
    )
    print(json.dumps({'reference_audit': reference_summary}, sort_keys=True), flush=True)
    measures=[]
    for name,fn in [('exhaustive',exhaustive),('extended',extended_geometry),('fiber',fiber),('parity',parity),('decoding',decoding)]:
        if args.part not in ('all',name):continue
        wall=time.monotonic();cpu=time.process_time()
        detail=fn(out)
        measures.append(dict(part=name,cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-wall,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,details=detail))
        print(json.dumps(measures[-1],sort_keys=True),flush=True)
    record=dict(arithmetic='exact fractions for every scientific equality',workers=1,unit_tests=testresult.testsRun,
        reference_audit=reference_summary, parts=measures,measured_cpu_seconds=sum(x['cpu_seconds'] for x in measures),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (out/('resources.json' if args.part=='all' else f'resources_{args.part}.json')).write_text(json.dumps(record,indent=2)+'\n')
    print('All requested checks passed. General theorems remain mathematical proofs, not machine-checked proofs.')

if __name__=='__main__':main()
