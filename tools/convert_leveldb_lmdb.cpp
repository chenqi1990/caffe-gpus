/*************************************************************************
    > File Name: convert_leveldb_lmdb.cpp
    > Author: ma6174
    > Mail: ma6174@163.com 
    > Created Time: Thu 05 Nov 2015 09:53:59 PM CST
 ************************************************************************/
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <lmdb.h>
#include <sys/stat.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using std::string;

int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
	namespace gflags = google;
#endif
	if (argc < 3) {
		std::cout << "Usage:" << argv[0] << " input output" << std::endl;
		return -1;
	}
	string input(argv[1]);
	string output(argv[2]);
	
	// leveldb
	leveldb::DB* db;
    leveldb::Options options;
	options.max_open_files = 100;
	options.create_if_missing = false;
	LOG(INFO) << "Opening leveldb " << input;
	leveldb::Status status = leveldb::DB::Open(options, input, &db);
	leveldb::Iterator* iter = db->NewIterator(leveldb::ReadOptions());
	iter->SeekToFirst();
	
	// lmdb
	MDB_env *mdb_env;
	MDB_dbi mdb_dbi;
	MDB_val mdb_key, mdb_data;
	MDB_txn *mdb_txn;
	LOG(INFO) << "Opening lmdb " << output;
    CHECK_EQ(mkdir(output.c_str(), 0744), 0) 
		<< "mkdir " << output.c_str() << "failed";
	CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
	CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS)  // 1TB
		<< "mdb_env_set_mapsize failed";
	CHECK_EQ(mdb_env_open(mdb_env, output.c_str(), 0, 0664), MDB_SUCCESS)
		<< "mdb_env_open failed";
	CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
		<< "mdb_txn_begin failed";
	CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS)
		<< "mdb_open failed. Does the lmdb already exist?";

	int count = 0;
	while (iter->Valid()) {
		string key = iter->key().ToString();
		string val = iter->value().ToString();

		mdb_data.mv_size = val.size();
		mdb_data.mv_data = reinterpret_cast<void*>(&val[0]);
		mdb_key.mv_size = key.size();
		mdb_key.mv_data = reinterpret_cast<void*>(&key[0]);
		CHECK_EQ(mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0), MDB_SUCCESS)
			<< "mdb_put failed";

		if (++count % 1000 == 0) {
			CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS)
				<< "mdb_txn_commit failed";
			CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
				<< "mdb_txn_begin failed";
		}
		iter->Next();
	}
	if (count % 1000 != 0) {
		CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS)
			<< "mdb_txn_commit failed";
	}

	mdb_close(mdb_env, mdb_dbi);
	mdb_env_close(mdb_env);
	delete db;
	return 0;
}
