#ifndef CAFFE_SOLVER_MULTI_GPU_HPP
#define CAFFE_SOLVER_MULTI_GPU_HPP

#include "caffe/solver.hpp"
#include "caffe/internal_thread.hpp"
#include <queue>
#include <vector>

namespace caffe {

const int BUFFER_SIZE = 2;

template <typename Dtype>
class SGDSolver_Advance : public SGDSolver<Dtype> {
public:
	virtual ~SGDSolver_Advance() {}
	explicit SGDSolver_Advance(  const SolverParameter& param,
		const int gpu_idx, const int gpu_num )
		: SGDSolver<Dtype>( param ) ,model_id_(0),
		gpu_idx_(gpu_idx), gpu_num_(gpu_num) {}
	// rewrite the base class's function Solve
	// to support the function of push_diff and updat_model
	virtual void Solve( const char* resume_file = NULL );
	virtual void Solve( const string resume_file ) { Solve( resume_file.c_str() ); }

	// update global_model_  with diffs from all gpus
	static void update_global_model(bool& all_finish_flag);
	// initial global_model_ with current gpu's model
	void initial_global_model( );
	inline static long long get_global_model_id() { return global_model_id_; }

	// update the model of current gpu with the global_model_ and global_state_
	void update_model(); 
	// push the diff of current gpu to the global diff queue
	void push_diff();

	inline void CopyTrainedLayersFrom( const char* finetuninig_file ) {
		this->net()->CopyTrainedLayersFrom( finetuninig_file );
	}

	inline void CopyTrainedLayersFrom( const string& finetuning_file ) {
		CopyTrainedLayersFrom( finetuning_file.c_str() );
	}

private:
	long long model_id_;
	int gpu_idx_;
	int gpu_num_;

	// attention, they are static values.

	// weight, bias
	static std::vector< shared_ptr< Blob<Dtype> > > global_params_vector_;
	// history(diff)
	static std::vector< shared_ptr< Blob<Dtype> > > global_diff_;
	// shared parameters
	//static shared_ptr< std::vector<int> > param_owners_ptr_;
	// history(diff) queue
	static std::vector< shared_ptr< Blob<Dtype> > > global_diff_buffer_[BUFFER_SIZE];
	// the number of times of global_params_'s update
	static long long global_model_id_; 
	// total iter across all gpus
	static int global_total_iter_;
};

template <typename Dtype>
class SGDSolver_Thread : 
	public InternalThread {
public:
	virtual ~SGDSolver_Thread() {
		JoinSolveThread();
	}
	explicit SGDSolver_Thread( const SolverParameter& param, 
		const int gpu_idx, const int gpu_num )
		: param_( param ), gpu_idx_(gpu_idx), gpu_num_(gpu_num){}
		
	virtual void InternalThreadEntry();
	virtual void CreateSolveThread();
	virtual void JoinSolveThread();

	void Solve_Thread( const int gpu_idx, const char* resume_file, 
		const char* finetuninig_file );

	inline const shared_ptr< SGDSolver_Advance<Dtype> >&
		get_sgd_solver() { return sgd_solver_advance_;}
private:
	shared_ptr< SGDSolver_Advance<Dtype> > sgd_solver_advance_;
	SolverParameter param_;
	string resume_file_;
	string finetuning_file_;
	int gpu_idx_;
	int gpu_num_;
};

template <typename Dtype>
class Update_Thread : 
	public InternalThread {
public:
	explicit Update_Thread() : all_finish_flag_(false) {}
	virtual void InternalThreadEntry();
	bool all_finish_flag_;
};
/**
 * @brief Optimizes the parameters of a Net using
 *        asynchronous stochastic gradient descent (ASGD) with multi-gpu.
 */
 template <typename Dtype>
 class ASGDSolver {
public:
	explicit ASGDSolver( const SolverParameter& param);
	explicit ASGDSolver( const string& param_file ) ;
	virtual ~ASGDSolver();
	void Init( const SolverParameter& param );

	void Resuming( const char* resume_file );
	inline void Resuming( const string& resume_file ) { Resuming( resume_file.c_str() ); }

	void Finetuning( const char* finetuninig_file );
	inline void Finetuning( const string& finetuninig_file ) { Finetuning( finetuninig_file.c_str() ); }
	void Solve();
private:
	// run the other models which have the same initialization 
	// parameters with the first model.
	// this function should be called after function "Solve".
	void SolveOthers();
	void JoinAllSolveThread();
	void update_global_model();
protected:
	SolverParameter param_;
	int gpu_num_; // the number of GPU card  
	int gpu_start_idx_;
	int gpu_end_idx_;
	// sgd_solver_vec_ stores all the model replicas in all GPU
	vector< shared_ptr<SGDSolver_Thread<Dtype> > > sgd_solver_vec_;
	DISABLE_COPY_AND_ASSIGN(ASGDSolver);
 };
} // namespace caffe

#endif // CAFFE_SOLVER_MULTI_GPU_HPP