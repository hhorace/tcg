/**
 * Framework for NoGo and similar games (C++ 11)
 * agent.h: Define the behavior of variants of the player
 *
 * Author: Theory of Computer Games
 *         Computer Games and Intelligence (CGI) Lab, NYCU, Taiwan
 *         https://cgilab.nctu.edu.tw/
 */

#pragma once
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include <fstream>
#include "board.h"
#include "action.h"

#include <chrono>
#include <cmath>
#include <map>
#include <tuple>

#include <thread>
#include<cstring>
#include <omp.h>
int best_move_[4] = { 0 };
int visits_move_[4][100] = { 0 };
board child_state;
size_t num_threads = 2;
// double threshold_time = 9.;
double remain_time = 300.;

class agent {
public:
	agent(const std::string& args = "") {
		std::stringstream ss("name=unknown role=unknown " + args);
		for (std::string pair; ss >> pair; ) {
			std::string key = pair.substr(0, pair.find('='));
			std::string value = pair.substr(pair.find('=') + 1);
			meta[key] = { value };
		}
	}
	virtual ~agent() {}
	virtual void open_episode(const std::string& flag = "") {}
	virtual void close_episode(const std::string& flag = "") {}
	virtual action take_action(const board& b) { return action(); }
	virtual bool check_for_win(const board& b) { return false; }

public:
	virtual std::string property(const std::string& key) const { return meta.at(key); }
	virtual void notify(const std::string& msg) { meta[msg.substr(0, msg.find('='))] = { msg.substr(msg.find('=') + 1) }; }
	virtual std::string name() const { return property("name"); }
	virtual std::string role() const { return property("role"); }
	virtual std::string cycle() const { return property("T"); }
	virtual std::string exp_cons() const { return property("exp"); }

protected:
	typedef std::string key;
	struct value {
		std::string value;
		operator std::string() const { return value; }
		template<typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
		operator numeric() const { return numeric(std::stod(value)); }
	};
	std::map<key, value> meta;
};

/**
 * base agent for agents with randomness
 */
class random_agent : public agent {
public:
	random_agent(const std::string& args = "") : agent(args) {
		if (meta.find("seed") != meta.end())
			engine.seed(int(meta["seed"]));
	}
	virtual ~random_agent() {}

protected:
	std::default_random_engine engine;
};

/**
 * random player for both side
 * put a legal piece randomly
 */
class player : public random_agent {
public:
	player(const std::string& args = "") : random_agent("name=random role=unknown " + args),
		space(board::size_x * board::size_y), who(board::empty) {
		if (name().find_first_of("[]():; ") != std::string::npos)
			throw std::invalid_argument("invalid name: " + name());
		if (role() == "black"){
			who = board::black;
			printf("black using random \n\n");
		}
		if (role() == "white"){
			who = board::white;
			printf("white using random \n\n");
		}
		if (who == board::empty)
			throw std::invalid_argument("invalid role: " + role());
		for (size_t i = 0; i < space.size(); i++)
			space[i] = action::place(i, who);
	}

	virtual action take_action(const board& state) {
		std::shuffle(space.begin(), space.end(), engine);
		for (const action::place& move : space) {
			board after = state;
			if (move.apply(after) == board::legal)
				return move;
		}
		return action();
	}

private:
	std::vector<action::place> space;
	board::piece_type who;
};

class MCTS_player : public random_agent {
public:
	MCTS_player(const std::string& args = "") : random_agent("name=mcts role=unknown " + args),
		space_size(board::size_x * board::size_y), who(-1) {
		if (name().find_first_of("[]():; ") != std::string::npos)
			throw std::invalid_argument("invalid name: " + name());
		try {
			cycles = std::stoi(cycle());
		} catch(std::exception &e) {}
		try{
			exploration_constant = std::stod(exp_cons());
		} catch(std::exception &e) {}

		if (role() == "black"){
			who = 1;
			printf("black using mcts with cycles(%d) exp(%.2f)\n\n", cycles, exploration_constant);
		}
		else if (role() == "white"){
			who = 0;
			printf("white using mcts with cycles(%d) exp(%.2f)\n\n", cycles, exploration_constant);
		}
		if (who == (size_t) -1) throw std::invalid_argument("invalid role: " + role());
		
	}
	void MCTS_child(size_t input){
		size_t thread_id = input;
		// constexpr const double threshold_time[36] = {
		// 	6.960377358, 7.07136515, 7.189148112, 7.314449136, 7.448103561, 7.591082714, 7.744523755, 7.909767954, 8.088410331, 8.282364912, 8.493951728, 8.726014687, 8.982084159, 9.266605794, 9.585270026, 9.945499158, 10.35718959, 10.83388378, 11.39470048, 12.06768051, 11.34983514, 10.67984613, 10.05452306, 9.470888185, 8.926162306, 8.417751486, 7.94323472, 7.500352405, 7.086995578, 6.701195873, 6.341116148, 6.005041738, 5.691372289, 5.398614137, 5.125373194, 4.870348315
		// };
		board b = board(child_state);
		Node root(engine, b, 1-who, space_size, nullptr, exploration_constant);

		constexpr const double threshold_time = 11.0;
		// constexpr const double threshold_time = 1.;
		const auto start_time = std::chrono::high_resolution_clock::now();
		double dt;
		int itr = 0;

		do {
			Node *node = &root;
			board b = board(child_state);
			// std::array<board::board_t, 2> rave;
			
			// selection
			while (!node->has_untried_moves() && node->has_children()) {
				node = node->get_UCT_child();
				auto &&[bw, pos] = node->get_move();
				action::place move = action::place(pos, (bw==1) ? board::black : board::white);
				move.apply(b);
				// rave[bw].set(pos);
			}
			// expansion
			if (node->has_untried_moves()) {
				auto &&[bw, pos] = node->pop_untried_move();
				action::place move = action::place(pos, (bw==1) ? board::black : board::white);
				move.apply(b);
				// rave[bw].set(pos);
				// std::cerr << b;
				node = node->add_child(engine, b, bw, pos, exploration_constant);
				// printf("node move_size: %d\n", node->moves_.size());
			}
			// simulation & rollout
			size_t bw = 1 - node->get_player();
			while (true) {
				std::vector<size_t> moves;
				for (size_t i = 0; i < board::size_x * board::size_y ; i++){
					action::place m = action::place(i, (bw==1) ? board::black : board::white);
					board tmp = board(b);
					if (m.apply(tmp) == board::legal) {
						moves.push_back(i);
					}
				}
				if (moves.empty()) break;

				std::uniform_int_distribution<size_t> choose(0, moves.size() - 1);
				auto it = moves.begin() + choose(engine);
				size_t pos = *it;
				moves.erase(it);

				action::place move = action::place(pos, (bw==1) ? board::black : board::white);
				move.apply(b);
				// if (is_two_go) {
				// rave[bw].set(pos);
					// }
				// std::cerr << b;
				bw = 1 - bw;
			};
			// size_t winner = playout(b, 1-node->get_player(), rave);
			size_t winner =  1 - bw;

			// backpropogation
			while (node != nullptr) {
				// node->update(winner == node->get_player(), rave);
				node->update(winner == node->get_player());
				node = node->get_parent();
			}
			// time threshold
			dt = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(
										std::chrono::high_resolution_clock::now() - start_time
									).count();
			// iteration threshold
			itr++;
		} while (dt < threshold_time/*[num_steps] /*&& itr<=cycles*/);
		// printf("costing: %f\n",dt);

		std::vector<Node> &children = root.get_children();
		if(children.empty()){
			best_move_[thread_id] = -1;
			return;
		}

		int best_move = -1;
		int max_visits = -1;
		for (const auto &child : children) {
			auto &&[wins, visits] = child.get_wins_visits();
			auto &&[bw, pos] = child.get_move();
			visits_move_[thread_id][pos] = visits;
			best_move = ((int) visits > max_visits) ? pos : best_move;
			max_visits = ((int) visits > max_visits) ? visits : max_visits;
			// printf("now visits(%d), max_visits(%d), best_move(%d)\n", visits, max_visits, best_move);
		}
		
		best_move_[thread_id] = best_move;
		return;
	}
	virtual void open_episode(const std::string& flag = "") {
		num_steps = 0;
		remain_time = 300;
	}
	virtual action take_action(const board& state) {
		const auto start_time = std::chrono::high_resolution_clock::now();
		num_steps++;
		if (num_steps <= 7){
			std::vector<int> vec = {3,	5,	35,	53,	77,	75,	27,	45,	30,
									32,	48,	50,	12,	13,	14,	15,	16,	17,
									18,	19,	20,	21,	22,	23,	24,	25,	26,
									6,	28,	29,	8,	31,	9,	33,	34,	2,
									36,	37,	38,	39,	40,	41,	42,	43,	44,
									7,	46,	47,	10,	49,	11,	51,	52,	0,
									54,	55,	56,	57,	58,	59,	60,	61,	62,
									63,	64,	65,	66,	67,	68,	69,	70,	71,
									72,	73,	74,	1,	76,	4,	78,	79,	80
			};
			// std::shuffle(vec.begin(), vec.end(), engine);
			
			for (const int i : vec) {
				auto move = action::place(i, (who==1) ? board::black : board::white);
				board after = state;
				if (move.apply(after) == board::legal){
					// double dt = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(
					// 					std::chrono::high_resolution_clock::now() - start_time
					// 				).count();
					// remain_time -= dt;
					// printf("dt: %f\n",dt);
					return move;
				}
			}
		}

		// threshold_time = remain_time / (15.0f + std::max(45.0f-num_steps*2, 0.0f)) + 2.8f;
		// threshold_time = remain_time / (5.0f + std::max(45.0f-num_steps*2, 0.0f));

		memset(best_move_, 0, sizeof(best_move_));
		memset(visits_move_, 0, sizeof(visits_move_));
		child_state = state;

		// pthread_create(&t, NULL, MCTS_child, (void*) &thread_id);
		std::thread threads[num_threads];
		for (size_t i = 0; i < num_threads; i++) {
			// size_t input[2] = {i, num_steps};
        	threads[i] = std::thread(&MCTS_player::MCTS_child, this, (size_t)i);
		}
		for (size_t i = 0; i < num_threads; i++) {
			threads[i].join();
		}
		// std::thread t1(&MCTS_player::MCTS_child, this, 0);
		// t1.join();
		// printf("best_move: %d, %d, %d, %d\n", best_move_[0],best_move_[1],best_move_[2],best_move_[3]);

		if(best_move_[0]==-1 && best_move_[1]==-1 && best_move_[2]==-1 && best_move_[3]==-1)	return action();
		else{
			int visits_move_sum[100] = {0};
			#pragma omp parallel for num_threads(num_threads)
			for(size_t i=0; i<100;i++){
				visits_move_sum[i] = visits_move_[0][i] + visits_move_[1][i] + visits_move_[2][i] + visits_move_[3][i];
			}
			
			// int sort_visits_move_sum[100];
			// memcpy(sort_visits_move_sum, visits_move_sum, 100*sizeof(int));
			// std::sort(sort_visits_move_sum, sort_visits_move_sum+100, std::greater<int>());
			// printf("%d, %d, %d, %d, %d\n", sort_visits_move_sum[0], sort_visits_move_sum[1], sort_visits_move_sum[2], sort_visits_move_sum[3], sort_visits_move_sum[4]);
			
			// double dt = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(
			// 							std::chrono::high_resolution_clock::now() - start_time
			// 						).count();
			// remain_time -= dt;
			// printf("dt: %f\n",dt);
			return action::place(
				std::distance(visits_move_sum, std::max_element(visits_move_sum, visits_move_sum + 100)), 
				(who==1) ? board::black : board::white
			);
		}
		// return action::place(best_move, (who==1) ? board::black : board::white);

		/*
		board b = board(state);
		// std::cerr << b;
		Node root(engine, b, 1-who, space_size, nullptr, exploration_constant);
		// printf("root move_size: %d\n", root.moves_.size());

		// random for first 30 steps
		// if(root.moves_.size()>40){
		// 	std::uniform_int_distribution<size_t> choose(0, root.moves_.size() - 1);
		// 	auto it = root.moves_.begin() + choose(engine);
		// 	size_t pos = *it;
		// 	return action::place(pos, (who==1) ? board::black : board::white);
		// }

		constexpr const double threshold_time = 7.;
		const auto start_time = std::chrono::high_resolution_clock::now();
    double dt;
		int itr = 0;
		
		do {
			Node *node = &root;
			board b = board(state);
			// std::array<board::board_t, 2> rave;
			
      // selection
			while (!node->has_untried_moves() && node->has_children()) {
        node = node->get_UCT_child();
				auto &&[bw, pos] = node->get_move();
				action::place move = action::place(pos, (bw==1) ? board::black : board::white);
				move.apply(b);
				// rave[bw].set(pos);
      }
      // expansion
      if (node->has_untried_moves()) {
				auto &&[bw, pos] = node->pop_untried_move();
        action::place move = action::place(pos, (bw==1) ? board::black : board::white);
        move.apply(b);
				// rave[bw].set(pos);
				// std::cerr << b;
        node = node->add_child(engine, b, bw, pos, exploration_constant);
				// printf("node move_size: %d\n", node->moves_.size());
      }
      // simulation & rollout
			size_t bw = 1 - node->get_player();
			while (true) {
				std::vector<size_t> moves;
				for (size_t i = 0; i < board::size_x * board::size_y ; i++){
					action::place m = action::place(i, (bw==1) ? board::black : board::white);
					board tmp = board(b);
					if (m.apply(tmp) == board::legal) {
						moves.push_back(i);
					}
				}
				if (moves.empty()) break;

				std::uniform_int_distribution<size_t> choose(0, moves.size() - 1);
				auto it = moves.begin() + choose(engine);
				size_t pos = *it;
				moves.erase(it);

				action::place move = action::place(pos, (bw==1) ? board::black : board::white);
				move.apply(b);
				// if (is_two_go) {
				// rave[bw].set(pos);
					// }
				// std::cerr << b;
				bw = 1 - bw;
			};
			// size_t winner = playout(b, 1-node->get_player(), rave);
			size_t winner =  1 - bw;

      // backpropogation
      while (node != nullptr) {
        // node->update(winner == node->get_player(), rave);
				node->update(winner == node->get_player());
        node = node->get_parent();
      }
			// time threshold
			dt = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(
										std::chrono::high_resolution_clock::now() - start_time
									).count();
			// iteration threshold
			itr++;
		} while (dt < threshold_time && itr<=cycles);
		printf("costing: %f\n",dt);
		
		std::vector<Node> &children = root.get_children();
		if(children.empty()) return action();

		std::map<size_t, float> move_ratio;
		for (const auto &child : children) {
      auto &&[wins, visits] = child.get_wins_visits();
			auto &&[bw, pos] = child.get_move();
			// float ratio = (float) wins / (float) visits;
			float ratio = (float) visits;
      move_ratio.emplace(pos, ratio);
			// printf("move(%d) ratio(%.1f=%d/%d) \n",pos,ratio,wins,visits);
    }
		size_t best_move = std::max_element(std::begin(move_ratio), std::end(move_ratio),
                                        [](const std::pair<size_t, float> p1, const std::pair<size_t, float> p2) {
                                          return p1.second < p2.second;
                                        })
                           ->first;

    return action::place(best_move, (who==1) ? board::black : board::white);
		*/
	}

private:
	std::vector<action::place> legal_moves;
	size_t space_size;
	size_t who;
	int cycles = 1000;
	double exploration_constant=0.25;
	size_t num_steps = 0;

	// size_t playout(board b, size_t bw, const std::array<board::board_t, 2> &rave) {
	// 	// const auto init_two_go = board.get_two_go();
	// 	// bool is_two_go;
  //   while (true) {
	// 		std::vector<size_t> moves;
	// 		for (size_t i = 0; i < board::size_x * board::size_y ; i++){
	// 			action::place m = action::place(i, (bw==1) ? board::black : board::white);
	// 			board tmp = board(b);
	// 			if (m.apply(tmp) == board::legal) {
	// 				moves.push_back(i);
	// 			}
	// 		}
  //     if (moves.empty()) break;

  //     std::uniform_int_distribution<size_t> choose(0, moves.size() - 1);
  //     auto it = moves.begin() + choose(engine);
  //     size_t pos = *it;
  //     moves.erase(it);

	// 		action::place move = action::place(pos, (bw==1) ? board::black : board::white);
	// 		move.apply(b);
	// 		// if (is_two_go) {
	// 		rave[bw].set(pos);
  //       // }
	// 		// std::cerr << b;
  //     bw = 1 - bw;
  //   };
  //   return 1-bw;
  // }

	class Node {
	public:	
		// Node() = default;
		Node(std::default_random_engine engine, board &b, 
				size_t who, size_t pos = board::size_x * board::size_y, Node *parent = nullptr, double exploration_constant=0.25){
			engine_ = engine;
			bw_ = who;
			pos_ = pos;
			parent_ = parent;
			exploration_constant_ = exploration_constant;
			// list all move that opponent can place
			size_t bw = 1-who;
			for (size_t i = 0; i < board::size_x * board::size_y ; i++){
				action::place m = action::place(i, (bw==1) ? board::black : board::white);
				board tmp = board(b);
				if (m.apply(tmp) == board::legal) {
					moves_.push_back(i);
					// if(parent==nullptr){
					// 	printf("pushing %d\n",i);
					// 	std::cerr << tmp;
					// }
				}
			}
		}
		// Node(const Node &) = default;
		// Node(Node &&) noexcept = default;
		// Node &operator=(const Node &) = default;
		// Node &operator=(Node &&) noexcept = default;
		~Node() = default;

	public:
		Node *get_parent() const { return parent_; };
		size_t get_player() const { return bw_; }
		std::vector<Node> &get_children() { return children_;	}

		std::tuple<size_t, size_t> get_wins_visits() const { return std::make_tuple(wins_, visits_); }

		std::tuple<size_t, size_t> get_move() const { return std::make_tuple(bw_, pos_); }

		Node *get_UCT_child() {
			for (auto &child : children_) {
				// child.uct_score_ =
				// 		double(child.wins_) / double(child.visits_) +
				// 		std::sqrt(/*2.0 * */std::log(double(visits_)) / child.visits_)*exploration_constant_;

				double avg = double(child.wins_) / double(child.visits_);
				double var = avg * (1.0 - avg);
				double first_term = std::sqrt(std::log(double(visits_)/double(child.visits_)));
				double second_term = std::min(0.25, var + sqrt(2.0 * log(double(visits_) / double(child.visits_))) );
				child.uct_score_ = avg + exploration_constant_ * first_term * second_term;
				
				// rave
				// child.uct_score_ = (child.rave_wins_ + child.wins_ + std::sqrt(std::log(double(visits_)) * child.visits_) * exploration_constant_) /
        //                     (child.rave_visits_ + child.visits_);
			}
			return &*std::max_element(children_.begin(), children_.end(),
																[](const Node &lhs, const Node &rhs) {
																	return lhs.uct_score_ < rhs.uct_score_;
																});
		}

		bool has_untried_moves() const { return !moves_.empty(); }
		bool has_children() const { return !children_.empty(); }

		std::tuple<size_t, size_t> pop_untried_move() {
			std::uniform_int_distribution<size_t> choose(0, moves_.size() - 1);
			auto it = moves_.begin() + choose(engine_);
			size_t pos = *it;
			moves_.erase(it);
			return std::make_tuple(1-bw_, pos);
		}
		
		
		Node *add_child(std::default_random_engine engine, board &b,
				size_t who, size_t pos = board::size_x * board::size_y, double exploration_constant=0.25) {
			Node node(engine, b, who, pos, this, exploration_constant);
			children_.emplace_back(node);
			return &children_.back();
		}
		// void update(bool win, const std::array<board::board_t, 2> &rave) {
		void update(bool win){
			++visits_;
			wins_ += win ? 1 : 0;

			// rave
			// const size_t csize = children_.size(),
      //              cwin = win ? 0 : 1;
      // const auto &rave_ = rave[1 - bw_];
      // for (size_t i = 0; i < csize; ++i) {
      //   auto &child = children_[i];

      //   if (rave_.BIT_TEST(child.pos_)) {
      //     ++child.rave_visits_;
      //     child.rave_wins_ += cwin;
      //   }
      // }
		}

	private:
	public:
		std::default_random_engine engine_;
		std::vector<Node> children_;
		std::vector<size_t> moves_;
		size_t bw_;
		size_t pos_;
		Node *parent_;
		double exploration_constant_ = 0.25;
		size_t visits_ = 0, wins_ = 0, rave_wins_ = 0, rave_visits_ = 0;;
		double uct_score_;
	};
};

