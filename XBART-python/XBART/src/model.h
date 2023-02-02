
#ifndef model_h
#define model_h

#include "common.h"
#include "utility.h"
#include <memory>
#include "state.h"
#include "X_struct.h"
#include "cdf.h"

using namespace std;

class tree;

class Model
{

public:
    size_t dim_theta;

    size_t dim_suffstat;

    size_t dim_residual;

    size_t class_operating;

    double no_split_penalty;

    // tree prior
    double alpha;

    double beta;

    Model(size_t dim_theta, size_t dim_suff)
    {
        this->dim_theta = dim_theta;
        this->dim_suffstat = dim_suff;
    };

    virtual ~Model() = default;

    // Abstract functions
    virtual void incSuffStat(State &state, size_t index_next_obs, std::vector<double> &suffstats) { return; };

    virtual void samplePars(State &state, std::vector<double> &suff_stat, std::vector<double> &theta_vector, double &prob_leaf) { return; };

    virtual void update_state(State &state, size_t tree_ind, X_struct &x_struct) { return; };

    virtual void initialize_root_suffstat(State &state, std::vector<double> &suff_stat) { return; };

    virtual void updateNodeSuffStat(State &state, std::vector<double> &suff_stat, matrix<size_t> &Xorder_std, size_t &split_var, size_t row_ind) { return; };

    virtual void calculateOtherSideSuffStat(std::vector<double> &parent_suff_stat, std::vector<double> &lchild_suff_stat, std::vector<double> &rchild_suff_stat, size_t &N_parent, size_t &N_left, size_t &N_right, bool &compute_left_side) { return; };

    virtual void state_sweep(size_t tree_ind, size_t M, matrix<double> &residual_std, X_struct &x_struct) const { return; };

    virtual double likelihood(std::vector<double> &temp_suff_stat, std::vector<double> &suff_stat_all, size_t N_left, bool left_side, bool no_split, State &state) const { return 0.0; };

    // virtual double likelihood_no_split(std::vector<double> &suff_stat, State&state) const { return 0.0; };

    virtual void ini_residual_std(State &state) { return; };

    // virtual double predictFromTheta(const std::vector<double> &theta_vector) const { return 0.0; };

    virtual void predict_std(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, matrix<double> &yhats_test_xinfo, vector<vector<tree>> &trees) { return; };

    virtual Model *clone() { return nullptr; };

    // Getters and Setters
    // num classes
    size_t getNumClasses() const { return dim_theta; };

    void setNumClasses(size_t n_class) { dim_theta = n_class; };

    // dim suff stat
    size_t getDimSuffstat() const { return dim_suffstat; };

    void setDimSuffStat(size_t dim_suff) { dim_suffstat = dim_suff; };

    // penalty
    double getNoSplitPenalty()
    {
        return no_split_penalty;
        ;
    };
    void setNoSplitPenalty(double pen) { this->no_split_penalty = pen; };

    virtual size_t get_class_operating() { return class_operating; };

    virtual void set_class_operating(size_t i)
    {
        class_operating = i;
        return;
    };
};

class NormalModel : public Model
{
public:
    size_t dim_suffstat = 3;

    // model prior
    // prior on sigma
    double kap;
    double s;
    double tau_kap;
    double tau_s;
    // prior on leaf parameter
    double tau; // might be updated if sampling tau
    double tau_prior;

    double tau_mean; // copy of the original value
    bool sampling_tau;

    NormalModel(double kap, double s, double tau, double alpha, double beta, bool sampling_tau, double tau_kap, double tau_s) : Model(1, 3)
    {
        this->kap = kap;
        this->s = s;
        this->tau_kap = tau_kap;
        this->tau_s = tau_s;
        this->tau_prior = tau;
        this->tau = tau;
        this->tau_mean = tau;
        this->alpha = alpha;
        this->beta = beta;
        this->dim_residual = 1;
        this->class_operating = 0;
        this->sampling_tau = sampling_tau;
    }

    NormalModel(double kap, double s, double tau, double alpha, double beta) : Model(1, 3)
    {
        this->kap = kap;
        this->s = s;
        this->tau = tau;
        this->tau_mean = tau;
        this->alpha = alpha;
        this->beta = beta;
        this->dim_residual = 1;
        this->class_operating = 0;
        this->sampling_tau = true;
    }

    NormalModel() : Model(1, 3) {}

    Model *clone() { return new NormalModel(*this); }

    void incSuffStat(State &state, size_t index_next_obs, std::vector<double> &suffstats);

    void samplePars(State &state, std::vector<double> &suff_stat, std::vector<double> &theta_vector, double &prob_leaf);

    void update_state(State &state, size_t tree_ind, X_struct &x_struct);

    void update_tau(State &state, size_t tree_ind, size_t sweeps, vector<vector<tree>> &trees);

    void update_tau_per_forest(State &state, size_t sweeps, vector<vector<tree>> &trees);

    void initialize_root_suffstat(State &state, std::vector<double> &suff_stat);

    void updateNodeSuffStat(State &state, std::vector<double> &suff_stat, matrix<size_t> &Xorder_std, size_t &split_var, size_t row_ind);

    void calculateOtherSideSuffStat(std::vector<double> &parent_suff_stat, std::vector<double> &lchild_suff_stat, std::vector<double> &rchild_suff_stat, size_t &N_parent, size_t &N_left, size_t &N_right, bool &compute_left_side);

    void state_sweep(size_t tree_ind, size_t M, matrix<double> &residual_std, X_struct &x_struct) const;

    double likelihood(std::vector<double> &temp_suff_stat, std::vector<double> &suff_stat_all, size_t N_left, bool left_side, bool no_split, State &state) const;

    // double likelihood_no_split(std::vector<double> &suff_stat, State&state) const;

    void ini_residual_std(State &state);

    void predict_std(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, matrix<double> &yhats_test_xinfo, vector<vector<tree>> &trees);

    void predict_whole_std(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, std::vector<double> &output_vec, vector<vector<tree>> &trees);
};


#endif