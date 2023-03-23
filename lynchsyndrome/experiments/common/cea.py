from typing import Optional, List

import numpy
import pandas

class CostEffectivenessAnalysis:

    def __init__(
        self,
        interventions: List[str],
        costs: numpy.array,
        effects: numpy.array,
        threshold: Optional[float] = None
    ):
        if len(interventions) != len(costs):
            raise ValueError('interventions and costs must have the same length')
        if len(interventions) != len(effects):
            raise ValueError('interventions and effects must have the same length')
        self.interventions = interventions
        self.costs = costs
        self.effects = effects
        self.threshold = threshold

        self.ce_frontier = self.identify_frontier()
        self.icers = self.calculate_icers(self.ce_frontier)
    
    def __str__(self):
        return str(self.as_pandas_df())
    
    def as_pandas_df(self):
        res = pandas.DataFrame({
            'Intervention': [self.interventions[i] for i in self.ce_frontier],
            'Costs': self.costs[self.ce_frontier],
            'Effects': self.effects[self.ce_frontier],
            'ICER': self.icers
        })
        if not self.threshold is None:
            ce = numpy.argwhere(self.icers <= self.threshold).flatten()
            
            res['CE'] = ['' for _ in self.ce_frontier]
            if len(ce) >= 1:
                res.loc[ce[-1], 'CE'] = '*'
            else:
                res.loc[0, 'CE'] = '*'
        return res


    def identify_frontier(self):
        """
        Calculate the cost-effectiveness frontier

        Returns a list of indices for the strategies on the cost-effectiveness
        frontier, sorted in order of ascending effects (equivalently sorted in
        order of ascending costs)

        As described in

            Suen S, Goldhaber-Fiebert JD. An efficient, non-iterative method
            of identifying the cost-effectiveness frontier. Med Decis Making
            2016; 36(1):132-136. DOI: 10.1177/0272989X15583496
        """
        N = len(self.interventions)
        
        # Create a matrix of pairwise ICERs
        icer_mat = numpy.empty((N, N))
        for i in range(1, N):
            for j in range(i):
                icer_mat[i,j] = (self.costs[j] - self.costs[i]) / (self.effects[j] - self.effects[i])
        
        # Pull out ICERs which are reasonable WTP
        all_icers = icer_mat[numpy.tril_indices(N, -1)]
        valid_icers = numpy.unique(
            all_icers[
                numpy.logical_and(all_icers > 0, numpy.isfinite(all_icers))
            ]
        )

        # Calculate NMBs and identify the frontier
        on_frontier = numpy.zeros((N,), dtype=numpy.bool8)
        for wtp in valid_icers:
            nmb = wtp * self.effects - self.costs
            highest_nmb = nmb == nmb.max()
            on_frontier = numpy.logical_or(on_frontier, highest_nmb)

        unsorted_frontier = numpy.flatnonzero(on_frontier)
        
        uf_effects = self.effects[unsorted_frontier]

        return unsorted_frontier[numpy.argsort(uf_effects)]

    def calculate_icers(self, ce_frontier):
        """
        Calculate the ICERs on the cost-effectiveness frontier
        
        Note - technically the instance (self) will have a member,
        self.ce_frontier, but it is feasible that a user will want
        to access this method and supply their own frontier for
        calculations (e.g., because an intervention has been removed
        from consideration in a scenario analysis)
        """
        icers = numpy.empty((len(ce_frontier),))
        icers[0] = numpy.nan
        for i in range(1, len(ce_frontier)):
            icers[i] = (
                (self.costs[ce_frontier[i]] - self.costs[ce_frontier[i-1]]) /
                (self.effects[ce_frontier[i]] - self.effects[ce_frontier[i-1]])
            )
        
        return icers
    
