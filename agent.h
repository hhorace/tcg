/**
 * Framework for Threes! and its variants (C++ 11)
 * agent.h: Define the behavior of variants of agents including players and environments
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
#include "weight.h"

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
 * base agent for agents with weight tables and a learning rate
 */
class weight_agent : public agent {
public:
	weight_agent(const std::string& args = "") : agent(args), alpha(0.1f) {
		if (meta.find("init") != meta.end())
			init_weights(meta["init"]);
		if (meta.find("load") != meta.end())
			load_weights(meta["load"]);
		if (meta.find("alpha") != meta.end())
			alpha = float(meta["alpha"]);
	}
	virtual ~weight_agent() {
		if (meta.find("save") != meta.end())
			save_weights(meta["save"]);
	}

protected:
	virtual void init_weights(const std::string& info) {
		// std::cout << "init_weights: " << info << std::endl;

		std::string res = info; // comma-separated sizes, e.g., "65536,65536"
		for (char& ch : res)
			if (!std::isdigit(ch)) ch = ' ';
		std::stringstream in(res);
		for (size_t size; in >> size; net.emplace_back(size));

		// for (int i=0; i < (int) net.size(); i++)	std::cout << net[i] << " " << std::endl;
	}
	virtual void load_weights(const std::string& path) {
		// std::cout << "load_weights: " << path << std::endl;

		std::ifstream in(path, std::ios::in | std::ios::binary);
		if (!in.is_open()) std::exit(-1);
		uint32_t size;
		in.read(reinterpret_cast<char*>(&size), sizeof(size));
		net.resize(size);
		for (weight& w : net) in >> w;
		in.close();
	}
	virtual void save_weights(const std::string& path) {
		// std::cout << "save_weights: " << path << std::endl;

		std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
		if (!out.is_open()) std::exit(-1);
		uint32_t size = net.size();
		out.write(reinterpret_cast<char*>(&size), sizeof(size));
		for (weight& w : net) out << w;
		out.close();
	}

	// accumulate the total value of given state
  float estimate(const board &b) const {
    float value = 0;
    for (auto& w : net) {
      value += w.estimate(b);
    }
    return value;
  }

	// update the value of given state and return its new value
  float update(const board &b, float u) {
    float u_split = u / net.size();
    float value = 0;
    for (weight& w : net) {
      value += w.update(b, u_split);
    }
    return value;
  }

protected:
	std::vector<weight> net;
	float alpha;
};

class td_slider : public weight_agent {
public:
  td_slider(const std::string &args = "")
      : weight_agent("name=slide role=td_slider " + args) {
		net.emplace_back(weight({0, 1, 2, 3, 4, 5}));
    net.emplace_back(weight({4, 5, 6, 7, 8, 9}));
    net.emplace_back(weight({0, 1, 2, 4, 5, 6}));
    net.emplace_back(weight({4, 5, 6, 8, 9, 10}));
    path_.reserve(20000);
  }

  virtual action take_action(const board &before) {
    board after[] = {board(before), board(before), board(before),
                     board(before)};
    int reward[] = {after[0].slide(0), after[1].slide(1),
                                after[2].slide(2), after[3].slide(3)};
		constexpr const float ninf = -std::numeric_limits<float>::max();
		// std::cout << "in take_action2" << estimate(after[0]) << std::endl;																			
    float value[] = {
        reward[0] == -1 ? ninf : reward[0] + estimate(after[0]),
        reward[1] == -1 ? ninf : reward[1] + estimate(after[1]),
        reward[2] == -1 ? ninf : reward[2] + estimate(after[2]),
        reward[3] == -1 ? ninf : reward[3] + estimate(after[3]),
    };
		// std::cout << "in take_action3" << std::endl;
    float *max_value = std::max_element(value, value + 4);
		// std::cout << "in take_action4" << std::endl;
    if (*max_value > ninf) {
      unsigned idx = max_value - value;
			// std::cout << "in take_action5" << std::endl;
      path_.emplace_back(state({.before = before,
                                .after = after[idx],
                                .op = idx,
                                .reward = static_cast<float>(reward[idx]),
                                .value = *max_value}));
			// std::cout << "in take_action6" << std::endl;																
      return action::slide(idx);
    }
		// std::cout << "in take_action7" << std::endl;
    path_.emplace_back(state());
		// std::cout << "in take_action8" << std::endl;
    return action();
  }

  void update_episode() {
    float exact = 0;
    for (path_.pop_back(); path_.size(); path_.pop_back()) {
      state &move = path_.back();
      float error = exact - (move.value - move.reward);
      exact = move.reward + update(move.after, alpha * error);
    }
    path_.clear();
  }

private:
  struct state {
    board before, after;
    unsigned op;
    float reward, value;
  };
  std::vector<state> path_;
};

/**
 * default random environment, i.e., placer
 * place the hint tile and decide a new hint tile
 */
class random_placer : public random_agent {
public:
	random_placer(const std::string& args = "") : random_agent("name=place role=placer " + args) {
		spaces[0] = { 12, 13, 14, 15 };
		spaces[1] = { 0, 4, 8, 12 };
		spaces[2] = { 0, 1, 2, 3};
		spaces[3] = { 3, 7, 11, 15 };
		spaces[4] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
	}

	virtual action take_action(const board& after) {
		std::vector<int> space = spaces[after.last()];
		std::shuffle(space.begin(), space.end(), engine);
		for (int pos : space) {
			if (after(pos) != 0) continue;

			int bag[3], num = 0;
			for (board::cell t = 1; t <= 3; t++)
				for (size_t i = 0; i < after.bag(t); i++)
					bag[num++] = t;
			std::shuffle(bag, bag + num, engine);

			board::cell tile = after.hint() ?: bag[--num];
			board::cell hint = bag[--num];

			return action::place(pos, tile, hint);
		}
		return action();
	}

private:
	std::vector<int> spaces[5];
};

/**
 * random player, i.e., slider
 * select a legal action randomly
 */
class random_slider : public random_agent {
public:
	random_slider(const std::string& args = "") : random_agent("name=slide role=slider " + args),
		opcode({ 0, 1, 2, 3 }) {} // URDL

	virtual action take_action(const board& before) {
		std::shuffle(opcode.begin(), opcode.end(), engine);
		for (int op : opcode) {
			// step += 1;
			// op = (op + step) % 4;
			// std::cerr << "Board.step: " << op << std::endl;
			board::reward reward = board(before).slide(op);
			if (reward != -1) return action::slide(op);
		}

		return action();
	}

private:
	std::array<int, 4> opcode;
	int step = 0;
};

class greedy_slider : public random_agent {
public:
	greedy_slider(const std::string& args = "") : random_agent("name=slide role=slider " + args),
		opcode({ 0, 1, 2, 3 }) {} // URDL

	virtual action take_action(const board& before) {
		float max_reward = -1;
		int max_op = 0;
		for (int op : opcode) {
			board now_board = board(before);
			int reward_1st_step = now_board.slide(op);
			// if (reward_1st_step == -1) continue;
			float this_reward = (float) reward_1st_step;
			for (int op2 : opcode){
				int reward_2nd_step = board(now_board).slide(op2);
				// if (reward_2nd_step == -1) continue;
				
				this_reward += (float) reward_2nd_step*0.2;
				// std::cerr << reward_1st_step << "+" << reward_2nd_step << "=" << this_reward << std::endl;
			}
			max_op = (this_reward > max_reward) ? op : max_op;
			max_reward = (this_reward > max_reward) ? this_reward : max_reward;
		}
		if (max_reward != -1) return action::slide(max_op);
		else return action();
	}

private:
	std::array<int, 4> opcode;
	int step = 0;
};
