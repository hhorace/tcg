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
		if (role() == "black") who = board::black;
		if (role() == "white") who = board::white;
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
		if (role() == "black") who = 1;
		if (role() == "white") who = 0;
		if (who == (size_t) -1) throw std::invalid_argument("invalid role: " + role());
	}

	virtual action take_action(const board& state) {
		board b = board(state);
		// std::cerr << b;
		Node root(engine, b, who, space_size);
		// printf("root move_size: %d\n", root.moves_.size());

		constexpr const double threshold_time = 1.;
		const auto start_time = std::chrono::high_resolution_clock::now();
    double dt;
		int itr = 0;
		do {
			Node *node = &root;
			board b = board(state);
			
      // selection
			while (!node->has_untried_moves() && node->has_children()) {
        node = node->get_UCT_child();
				auto &&[bw, pos] = node->get_move();
				action::place move = action::place(pos, (bw==1) ? board::black : board::white);
				move.apply(b);
				// printf("select\n");
      }
      // expansion
      if (node->has_untried_moves()) {
				auto &&[bw, pos] = node->pop_untried_move();
				if(node->get_parent() == nullptr) bw = 1-bw;
        action::place move = action::place(pos, (bw==1) ? board::black : board::white);
        move.apply(b);
				// std::cerr << b;
        node = node->add_child(engine, b, bw, pos);
				// printf("node move_size: %d\n", node->moves_.size());
      }
      // simulation & rollout
      size_t winner = playout(b, node->get_player()-1);
			// printf("winner: %d\n", winner);
      // backpropogation
      while (node != nullptr) {
        node->update(winner == node->get_player());
        node = node->get_parent();
      }
			// time threshold
			dt = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(
										std::chrono::high_resolution_clock::now() - start_time
									).count();
			// iteration threshold
			itr++;
		} while (dt < threshold_time && itr<=100);
		
		std::vector<Node> &children = root.get_children();
		if(children.empty()) return action();

		std::map<size_t, float> move_ratio;
		for (const auto &child : children) {
      auto &&[wins, visits] = child.get_wins_visits();
			auto &&[bw, pos] = child.get_move();
			float ratio = (float) wins / (float) visits;
      move_ratio.emplace(pos, ratio);
			// printf("move(%d) ratio(%.1f=%d/%d) \n",pos,ratio,wins,visits);
    }
		size_t best_move = std::max_element(std::begin(move_ratio), std::end(move_ratio),
                                        [](const std::pair<size_t, float> p1, const std::pair<size_t, float> p2) {
                                          return p1.second < p2.second;
                                        })
                           ->first;
    return action::place(best_move, (who==1) ? board::black : board::white);
	}

private:
	std::vector<action::place> legal_moves;
	size_t space_size;
	size_t who;

	size_t playout(board b, size_t bw) {
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
			// std::cerr << b;
      bw = 1 - bw;
    };
    return 1-bw;
  }

	class Node {
	public:	
		// Node() = default;
		Node(std::default_random_engine engine, board &b, 
				size_t who, size_t pos = board::size_x * board::size_y, Node *parent = nullptr){
			engine_ = engine;
			bw_ = who;
			pos_ = pos;
			parent_ = parent;
			for (size_t i = 0; i < board::size_x * board::size_y ; i++){
				action::place m = action::place(i, (who==1) ? board::black : board::white);
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
				child.uct_score_ =
						double(child.wins_) / double(child.visits_) +
						std::sqrt(2.0 * std::log(double(visits_)) / child.visits_);
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
				size_t who, size_t pos = board::size_x * board::size_y) {
			Node node(engine, b, who, pos, this);
			children_.emplace_back(node);
			return &children_.back();
		}
		void update(bool win) {
			++visits_;
			wins_ += win ? 1 : 0;
		}

	private:
	public:
		std::default_random_engine engine_;
		std::vector<Node> children_;
		std::vector<size_t> moves_;
		size_t bw_;
		size_t pos_;
		Node *parent_;
		size_t visits_ = 0, wins_ = 0;
		double uct_score_;
	};
};

