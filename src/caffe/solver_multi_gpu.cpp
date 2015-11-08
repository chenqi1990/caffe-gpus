#include <cstdio>

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "caffe/solver_multi_gpu.hpp"
#include "caffe/data_layers.hpp"
#include <unistd.h>
#include <pthread.h>
#include <semaphore.h>

namespace caffe {

template <typename Dtype> std::vector< shared_ptr< Blob<Dtype> > > 
	SGDSolver_Advance<Dtype>::global_params_vector_;
template <typename Dtype> std::vector< shared_ptr< Blob<Dtype> > >
	SGDSolver_Advance<Dtype>::global_diff_;
template <typename Dtype> std::vector< shared_ptr< Blob<Dtype> > >
	SGDSolver_Advance<Dtype>::global_diff_buffer_[BUFFER_SIZE];
template <typename Dtype> long long 
	SGDSolver_Advance<Dtype>::global_model_id_ = 0;
template <typename Dtype>  int 
	SGDSolver_Advance<Dtype>::global_total_iter_ = 0;
// producer-consumer-problem from 
// www.codingdevil.com/2014/04/c-program-for-producer-consumer-problem.html
pthread_mutex_t mutex;
sem_t full, empty;
int start  = 0;
int end = 0;
//----------------------------------------------------------------------------------------------------

template <typename Dtype>
void SGDSolver_Advance<Dtype>::initial_global_model() {
	LOG(INFO) << "initial global model";
	const std::vector<int>& param_owners = this->net()->param_owners();
	const std::vector<shared_ptr<Blob<Dtype> > >& params = this->net()->params();
	// allocate memory for global_params and global_diff
	global_params_vector_.clear();
	global_diff_.clear();
	LOG(INFO) << "allocate memory for global_params and global_diff";
	for (int i = 0; i < params.size(); ++i) {
		const Blob<Dtype>* net_param = params[i].get();
		global_params_vector_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(
			net_param->num(), net_param->channels(), net_param->height(),
			net_param->width())));
		global_diff_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(
			net_param->num(), net_param->channels(), net_param->height(),
			net_param->width())));
	}
	// deal with the shared params
	LOG(INFO) << "deal with the shared params";
	for (int i = 0; i < params.size(); ++i) {
		if ( param_owners[i] >= 0 ) {
			global_params_vector_[i]->ShareData(
				*( global_params_vector_[ param_owners[i] ] ) );
		}
	}
	// copy the value to global_params_vector_
	LOG(INFO) << "copy the value to global_params_vector_";
	for (int i = 0; i < params.size(); ++i) {
		if ( param_owners[i] >= 0 ) { continue; }
		memcpy( global_params_vector_[i]->mutable_cpu_data(),
			params[i]->cpu_data(), params[i]->count()*sizeof(Dtype)	);
	}
	// allocate memory for buffers
	LOG(INFO) << "allocate memory for buffers";
	for (int j=0; j<BUFFER_SIZE; j++) {
		for (int i = 0; i < params.size(); ++i) {
			const Blob<Dtype>* net_param = params[i].get();
			global_diff_buffer_[ j ].push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(
				net_param->num(), net_param->channels(), net_param->height(),
				net_param->width())));
		}
	}
	
	global_model_id_ = 1;
	LOG(INFO) << "initial global model finish";
}

//consumer process
template <typename Dtype>
void SGDSolver_Advance<Dtype>::update_global_model(bool& all_finish_flag) {
	do {
		sem_wait( &full );
		pthread_mutex_lock( &mutex );

		// get diff
		for (int i=0; i<global_params_vector_.size(); i++) {
			memcpy( 
				global_diff_[i]->mutable_cpu_data(),
				global_diff_buffer_[ start ][i]->cpu_data(), 
				global_diff_[i]->count()*sizeof(Dtype)	);
		}
		// add the diff to global_state_
		for (int i=0; i<global_params_vector_.size(); i++) {
			caffe_axpy<Dtype>( global_params_vector_[i]->count(), Dtype(-1),
				static_cast<const Dtype*>( global_diff_[i]->cpu_data() ),
				static_cast<Dtype*>( global_params_vector_[i]->mutable_cpu_data() ) );
		}
		global_model_id_++;
		start = (start+1)%BUFFER_SIZE;
		global_total_iter_ = global_model_id_-1;

		pthread_mutex_unlock( &mutex );
		sem_post( &empty );
	}while( all_finish_flag == false );
}

template <typename Dtype>
void SGDSolver_Advance<Dtype>::update_model() {
	//LOG(INFO) << "update model : " << model_id_ << " --> " << global_model_id_;
	if (model_id_ == global_model_id_)
		return;
	const std::vector<int>& param_owners = this->net()->param_owners();
	const std::vector<shared_ptr<Blob<Dtype> > >& params = this->net()->params();

	// copy the value to params
	for (int i = 0; i < params.size(); ++i) {
		if ( param_owners[i] >= 0 ) { continue; }
		memcpy( params[i]->mutable_cpu_data(),
			global_params_vector_[i]->cpu_data(), params[i]->count()*sizeof(Dtype) );
	}
	// copy the value to history
	std::vector<shared_ptr<Blob<Dtype> > >& history = this->history_;
	for (int i=0; i<history.size(); i++) {
		memcpy( history[i]->mutable_cpu_data(),
			global_diff_[i]->cpu_data(), history[i]->count()*sizeof(Dtype) );
	}
	model_id_ = global_model_id_;
}

// producer process
template <typename Dtype>
void SGDSolver_Advance<Dtype>::push_diff() {
	sem_wait( &empty );
	pthread_mutex_lock( &mutex );
	// copy the value
	std::vector<shared_ptr<Blob<Dtype> > >& history = this->history_;
	for (int i=0; i<history.size(); i++) {
		memcpy( global_diff_buffer_[ end ][i]->mutable_cpu_data(),
			history[i]->cpu_data(), history[i]->count()*sizeof(Dtype) );
	}
	end = (end+1)%BUFFER_SIZE;

	update_model();

	pthread_mutex_unlock( &mutex );
	sem_post( &full );

}

template <typename Dtype>
void SGDSolver_Advance<Dtype>::Solve( const char* resume_file ) {

	int current_device;
	CUDA_CHECK(cudaGetDevice(&current_device));
	this->set_total_iter( &global_total_iter_ );
	//( (DataLayer<Dtype>*)(this->net_->layer_by_name("data").get()) )->set_gpu_info(gpu_idx_, gpu_num_);

	Caffe::set_phase(Caffe::TRAIN);
	LOG(INFO) << "Solving " << this->net_->name();
	LOG(INFO) << "Learning Rate Policy: " << this->param_.lr_policy();
	this->PreSolve();

	this->iter_ = 0;
	this->current_step_ = 0;

	if (resume_file) {
		LOG(INFO) << "Restoring previous solver status from " << resume_file;
		this->Restore(resume_file);
	}
	// Remember the initial iter_ value; will be non-zero if we loaded from a
	// resume_file above.
	const int start_iter = this->iter_;

	int average_loss = this->param_.average_loss();

	CHECK_GE(average_loss, 1) << "average_cost should be non-negative.";

	vector<Dtype> losses;
	Dtype smoothed_loss = 0;

	// For a network that is trained by the solver, no bottom or top vecs
	// should be given, and we will just provide dummy vecs.
	vector<Blob<Dtype>*> bottom_vec;
	for (; this->iter_ < this->param_.max_iter(); ++this->iter_) {
		// Save a snapshot if needed.
		if (this->param_.snapshot() && this->iter_ > start_iter &&
			this->iter_ % this->param_.snapshot() == 0) {
			if (this->iter_ / this->param_.snapshot() % Caffe::gpu_num() == current_device){
				this->Snapshot();
			}
		}

		if (this->param_.test_interval() && this->iter_ % this->param_.test_interval() == 0
			&& (this->iter_ > 0 || this->param_.test_initialization())) {
			if ( this->iter_ / this->param_.test_interval() % Caffe::gpu_num() == current_device ) {
				this->TestAll();
			}
		}

		const bool display = this->param_.display() && this->iter_ % this->param_.display() == 0;
		this->net_->set_debug_info(display && this->param_.debug_info());

		Dtype loss = this->net_->ForwardBackward(bottom_vec);

		if (losses.size() < average_loss) {

			losses.push_back(loss);
			int size = losses.size();
			smoothed_loss = (smoothed_loss * (size - 1) + loss) / size;

		} else {

			int idx = (this->iter_ - start_iter) % average_loss;
			smoothed_loss += (loss - losses[idx]) / average_loss;
			losses[idx] = loss;
		}
		
		if (display) {
			// LOG(INFO) <<"Device " << current_device<< " Iteration " << this->iter_ << ", loss = " << smoothed_loss
			// 	<< "   diff_size = "<<global_diff_queue_.size();
			const vector<Blob<Dtype>*>& result = this->net_->output_blobs();
			int score_index = 0;
			for (int j = 0; j < result.size(); ++j) {
				const Dtype* result_vec = result[j]->cpu_data();
				const string& output_name =
				this->net_->blob_names()[this->net_->output_blob_indices()[j]];
				const Dtype loss_weight =
				this->net_->blob_loss_weights()[this->net_->output_blob_indices()[j]];
				for (int k = 0; k < result[j]->count(); ++k) {
					ostringstream loss_msg_stream;
					if (loss_weight) {
						loss_msg_stream << " (* " << loss_weight
							<< " = " << loss_weight * result_vec[k] << " loss)";
					}
					LOG(INFO) <<"Device " << current_device << "   Train net output #"
						<< score_index++ << ": " << output_name << " = "
						<< result_vec[k] << loss_msg_stream.str() 
						<< " TotIt = " << global_total_iter_ 
						<< " BufNum = " << abs( end-start );
				}
			}
		}

		this->ComputeUpdateValue();
		//this->net_->Update();
		push_diff();
	}
	// Always save a snapshot after optimization, unless overridden by setting
	// snapshot_after_train := false.
	if (this->param_.snapshot_after_train()) { this->Snapshot(); }
	// After the optimization is done, run an additional train and test pass to
	// display the train and test loss/outputs if appropriate (based on the
	// display and test_interval settings, respectively).  Unlike in the rest of
	// training, for the train net we only run a forward pass as we've already
	// updated the parameters "max_iter" times -- this final pass is only done to
	// display the loss, which is computed in the forward pass.
	if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
		Dtype loss;
		this->net_->Forward(bottom_vec, &loss);
		LOG(INFO) << "Device " << current_device << " Iteration " << this->iter_ << ", loss = " << loss;
	}
	if (this->param_.test_interval() && this->iter_ % this->param_.test_interval() == 0) {
		this->TestAll();
	}
	LOG(INFO)<<"Device " << current_device<< " Optimization Done.";
}

//----------------------------------------------------------------------------------------------------

template <typename Dtype>
void SGDSolver_Thread<Dtype>::CreateSolveThread() {
  CHECK(StartInternalThread()) << "Thread execution failed";
}

template <typename Dtype>
void SGDSolver_Thread<Dtype>::JoinSolveThread() {
  CHECK(WaitForInternalThreadToExit()) << "Thread joining failed";
}

template <typename Dtype>
void SGDSolver_Thread<Dtype>::InternalThreadEntry() 
{
	Caffe::SetDevice( gpu_idx_ );

	sgd_solver_advance_.reset( new SGDSolver_Advance<Dtype>( param_, gpu_idx_, gpu_num_ ) );
	if (SGDSolver_Advance<Dtype>::get_global_model_id() == 0) {
		sgd_solver_advance_->initial_global_model(); // update the global_model_

		if ( finetuning_file_.empty() == false ) {
			sgd_solver_advance_->CopyTrainedLayersFrom( finetuning_file_ );
		}
		if (resume_file_.empty() == true) {
			sgd_solver_advance_->Solve();
		} else {
			sgd_solver_advance_->Solve( resume_file_ );
		}

	} else {
		sgd_solver_advance_->update_model();
		sgd_solver_advance_->Solve();
	}
}

template <typename Dtype>
void SGDSolver_Thread<Dtype>::Solve_Thread( const int gpu_idx, const char* resume_file,
												const char* finetuning_file) {
	if ( resume_file == NULL ){
		resume_file_.clear();
	} else {
		resume_file_ = resume_file;
	}
	if (finetuning_file == NULL) {
		finetuning_file_.clear();
	} else {
		finetuning_file_ = resume_file;
	}
	gpu_idx_ = gpu_idx;	
	// First, join the thread
	JoinSolveThread();
	// Start a new prefetch thread
	CreateSolveThread();
}

//-----------------------------------------------------------------------------------------------------

template <typename Dtype>
ASGDSolver<Dtype>::ASGDSolver( const SolverParameter& param ) {
	Init( param );
}

template <typename Dtype>
ASGDSolver<Dtype>::ASGDSolver( const string& param_file ) {
	SolverParameter param;
	ReadProtoFromTextFileOrDie(param_file, &param);
	Init(param);
}

template <typename Dtype>
ASGDSolver<Dtype>::~ASGDSolver() {
}

template <typename Dtype>
void ASGDSolver<Dtype>::Init(const SolverParameter& param) {
	LOG(INFO) << "Initializing solver from parameters";
	param_ = param;
	gpu_num_ = Caffe::gpu_num();
	gpu_start_idx_ = Caffe::gpu_start_idx();
	gpu_end_idx_ = Caffe::gpu_end_idx();
	CHECK_EQ( gpu_num_, gpu_end_idx_-gpu_start_idx_+1 )
		 << "some thing wrong in gpu_num_, gpu_start_idx_ and gpu_end_idx_";
	
	sgd_solver_vec_.resize( gpu_num_ );
	
	for (int i=0; i<gpu_num_; i++) {
		sgd_solver_vec_[ i ].reset( new SGDSolver_Thread<Dtype>(param_, i, gpu_num_) );
	}

	CHECK_EQ( pthread_mutex_init( &mutex, NULL), 0 )
		<< "mutex init failed";
	sem_init( &full, 0, 0 );
	sem_init( &empty, 0, BUFFER_SIZE );

}

template <typename Dtype>
void ASGDSolver<Dtype>::Resuming( const char* resume_file ) {
	CHECK( resume_file != NULL ) << "resume_file should not be NULL.";
	sgd_solver_vec_[0]->Solve_Thread( gpu_start_idx_, resume_file, NULL );
	SolveOthers();
}

template <typename Dtype>
void ASGDSolver<Dtype>::Finetuning( const char* finetuning_file ) {
	CHECK( finetuning_file != NULL  ) << "finetuning_file should not be NULL.";
	sgd_solver_vec_[0]->Solve_Thread( gpu_start_idx_, NULL, finetuning_file );
	SolveOthers();
}

template <typename Dtype>
void ASGDSolver<Dtype>::Solve( ) {
	sgd_solver_vec_[0]->Solve_Thread( gpu_start_idx_, NULL, NULL );
	SolveOthers();
}

template <typename Dtype>
void ASGDSolver<Dtype>::SolveOthers() {
	// wait the first gpu to finish the initialization of global_model_
	while( SGDSolver_Advance<Dtype>::get_global_model_id() == 0 ) {sleep(0.01);}
	// run the other models which have the same initialization 
	// parameters with the global_model_.
	for (int i=1; i<gpu_num_; i++) {
		// wait the previous model to finish the data loading 
		// in consideration of bandwidth
		sgd_solver_vec_[i]->Solve_Thread( i+gpu_start_idx_, NULL, NULL );
	}
	// update_global_model
	Update_Thread<Dtype> update_thread;
	update_thread.all_finish_flag_ = false;
	update_thread.WaitForInternalThreadToExit();
	update_thread.StartInternalThread();
	JoinAllSolveThread();
	LOG(INFO) << "finish-----------------";
	update_thread.all_finish_flag_ = true;
	update_thread.WaitForInternalThreadToExit();
}

template <typename Dtype>
void ASGDSolver<Dtype>::JoinAllSolveThread() {
	for (int i=0; i<gpu_num_; i++) {
		sgd_solver_vec_[i]->JoinSolveThread();
	}
}

template <typename Dtype>
void Update_Thread<Dtype>::InternalThreadEntry() {
	SGDSolver_Advance<Dtype>::update_global_model( all_finish_flag_ );
}

INSTANTIATE_CLASS(SGDSolver_Advance);
INSTANTIATE_CLASS(ASGDSolver);
INSTANTIATE_CLASS(SGDSolver_Thread);
INSTANTIATE_CLASS(Update_Thread);
}  // namespace caffe