//////////////////////////////////////////////////////////////////////////////////////
// main function of the Bayesian backfitting algorithm
//////////////////////////////////////////////////////////////////////////////////////

#include "mcmc_loop.h"

void mcmc_loop(matrix<size_t> &Xorder_std, bool verbose, matrix<double> &sigma_draw_xinfo, vector<vector<tree>> &trees, double no_split_penalty, State &state, NormalModel *model, X_struct &x_struct, std::vector<double> &resid)
{
    size_t N = (*state.residual_std)[0].size();

    // initialize the matrix of residuals
    model->ini_residual_std(state);

    for (size_t sweeps = 0; sweeps < state.num_sweeps; sweeps++)
    {

        if (verbose == true)
        {
            COUT << "--------------------------------" << endl;
            COUT << "number of sweeps " << sweeps << endl;
            COUT << "--------------------------------" << endl;
        }

        for (size_t tree_ind = 0; tree_ind < state.num_trees; tree_ind++)
        {

            if (verbose)
            {
                COUT << "sweep " << sweeps << " tree " << tree_ind << endl;
            }

            // draw Sigma
            model->update_state(state, tree_ind, x_struct);

            sigma_draw_xinfo[sweeps][tree_ind] = state.sigma;

            if (state.use_all && (sweeps > state.burnin) && (state.mtry != state.p))
            {
                state.use_all = false;
            }

            // clear counts of splits for one tree
            std::fill((*state.split_count_current_tree).begin(), (*state.split_count_current_tree).end(), 0.0);

            // subtract old tree for sampling case
            if (state.sample_weights)
            {
                (*state.mtry_weight_current_tree) = (*state.mtry_weight_current_tree) - (*state.split_count_all_tree)[tree_ind];
            }

            // initialize sufficient statistics of the current tree to be updated
            model->initialize_root_suffstat(state, trees[sweeps][tree_ind].suff_stat);

            if (state.parallel)
            {
                trees[sweeps][tree_ind].settau(model->tau_prior, model->tau); // initiate tau
            }

            // main function to grow the tree from root
            trees[sweeps][tree_ind].grow_from_root(state, Xorder_std, x_struct.X_counts, x_struct.X_num_unique, model, x_struct, sweeps, tree_ind);

            // set id for bottom nodes
            tree::npv bv;
            trees[sweeps][tree_ind].getbots(bv); // get bottom nodes
            for (size_t i = 0; i < bv.size(); i++)
            {
                bv[i]->setID(i + 1);
            }

            // store residuals:
            for (size_t data_ind = 0; data_ind < (*state.residual_std)[0].size(); data_ind++)
            {
                resid[data_ind + sweeps * N + tree_ind * state.num_sweeps * N] = (*state.residual_std)[0][data_ind];
            }

            if (sweeps >= state.burnin)
            {
                for (size_t i = 0; i < (*state.split_count_all).size(); i++)
                {
                    (*state.split_count_all)[i] += (*state.split_count_current_tree)[i];
                }
            }

            // count number of splits at each variable
            state.update_split_counts(tree_ind);

            // update partial residual for the next tree to fit
            model->state_sweep(tree_ind, state.num_trees, (*state.residual_std), x_struct);
        }

        if (model->sampling_tau)
        {
            // update tau per sweep (after drawing a forest)
            model->update_tau_per_forest(state, sweeps, trees);
        }
    }
    return;
}
