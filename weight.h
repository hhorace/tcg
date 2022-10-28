/**
 * Framework for Threes! and its variants (C++ 11)
 * weight.h: Lookup table template for n-tuple network
 *
 * Author: Theory of Computer Games
 *         Computer Games and Intelligence (CGI) Lab, NYCU, Taiwan
 *         https://cgilab.nctu.edu.tw/
 */

#pragma once
#include <iostream>
#include <vector>
#include <utility>
#include "board.h"
#include <iterator>
#include <sstream>
#include <cassert>


/**
 * the pattern feature including isomorphism
 *
 * usage:
 *   weight({ 0, 1, 2, 3 })
 *   weight({ 0, 1, 2, 3, 4, 5 })
 *
 * isomorphic level of the pattern:
 *   1: no isomorphism
 *   4: enable rotation
 *   8: enable rotation and reflection
 */

class weight {
public:
	typedef float type;
	typedef uint64_t score;
	typedef uint32_t cell;
	typedef std::array<cell, 4> row;
	typedef std::array<row, 4> grid;
	typedef uint64_t data;

public:
  weight() = default;
  // there's 2^4 different kind of numbers in every tile, including, 0,1,2,3,6,12,24,48,96,...
  weight(const std::vector<score> &p) : value(1 << (p.size() << 2)) { 
  // weight(const std::vector<score> &p) : value(p.size()) {  
    size_t psize = p.size();
		assert(psize != 0);

    
    // std::cout << "value.size: " << value.size() << ", " << p.size() << std::endl;
    // for (int i=0;i<value.size();i++)
    //   std::cout << value[i] << " ";
    // std::cout << std::endl;

    row row1 = {0,1,2,3};
    row row2 = {4,5,6,7};
    row row3 = {8,9,10,11};
    row row4 = {12,13,14,15};
    grid b = {row1, row2, row3, row4};
    data v = 0;
    // board idx(b, v);
    // board idx;
    // for(int i=0; i<16; i++)
    //   idx.operator()(i) = i;
    // idx.reset();
    // for (int i=0; i<16; i++){
    //   std::cout << "#i=" << i << "? " << idx.place(i, i, i+1) << std::endl;
    //   std::cout << idx << std::endl;
    // }
    // std::cout << idx << std::endl;
    
    for (size_t i = 0; i < iso_level_; ++i) {
			board idx(b, v);
      // board idx(0xfedcba9876543210ull);
      if (i >= 4) {
        idx.reflect_horizontal();
      }
      idx.rotate(i);
      // std::cout << idx << std::endl;
      isomorphism[i].reserve(psize);
      for (auto t : p) {
        //  std::cout << idx(t) << " " << std::endl;
        isomorphism[i].push_back(idx(t));
      }
    }
    // std::cout << std::endl;
    // std::cout <<  "(" << isomorphism.size() << ")" <<std::endl;
    // for(int i=0; i<isomorphism.size();i++){
    //   std::cout << "(" << isomorphism[i].size() << ")";
    //   for(int j=0;j<isomorphism[i].size(); j++){
    //     std::cout << isomorphism[i][j] << " ";
    //   }
    //   std::cout << std::endl;
    // }
    // std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~leave~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  }
	weight(size_t len) : value(len) {}
	weight(weight&& f) : value(std::move(f.value)) {}
	weight(const weight& f) = default;

	weight& operator =(const weight& f) = default;
	type& operator[] (size_t i) { return value[i]; }
	const type& operator[] (size_t i) const { return value[i]; }
	size_t size() const { return value.size(); }

public:
	// estimate the value of a given board
  float estimate(const board &b) const {
    float value_sum = 0;

    // std::cout <<  "(" << isomorphism.size() << ")" <<std::endl;
    // for(int i=0; i<isomorphism.size();i++){
    //   std::cout << "(" << isomorphism[i].size() << ")";
    //   for(int j=0;j<isomorphism[i].size(); j++){
    //     std::cout << isomorphism[i][j] << " ";
    //   }
    //   std::cout << std::endl;
    // }

    for (size_t i = 0; i < iso_level_; ++i) {
      size_t index = indexof(isomorphism[i], b);
      value_sum += value[index];
    }
    return value_sum;
  }

  // update the value of a given board, and return its updated value
  float update(const board &b, float u) {
    float u_split = u / iso_level_;
    float value_sum = 0;
    for (size_t i = 0; i < iso_level_; ++i) {
      size_t index = indexof(isomorphism[i], b);
      value[index] += u_split;
      value_sum += value[index];
    }
    return value_sum;
  }

	
	size_t indexof(const std::vector<score> &p, const board &b) const {
    size_t index = 0;
    // std::cout << b;
    for (size_t i = 0; i < p.size(); ++i){
      index |= b(p[i]) << (i << 2); // b(p[i]) maps 0,1,2,3,6,12,24... to 0,1,2,3,4,5,6...
    }
    return index;
  }
	std::string nameof(const std::vector<score> &p) const {
    std::stringstream ss;
    ss << std::hex;
    std::copy(std::begin(p), std::end(p),
              std::ostream_iterator<score>(ss, ""));
    return ss.str();
  }
  std::string name() const {
    return std::to_string(isomorphism[0].size()) + "-tuple pattern " +
           nameof(isomorphism[0]);
  }
	friend std::ostream& operator <<(std::ostream& out, const weight& w) {
		std::string name = w.name();
    uint32_t len = name.length();
    out.write(reinterpret_cast<char *>(&len), sizeof(len));
    out.write(name.c_str(), len);
		// weight
		auto& value = w.value;
		uint64_t size = value.size();
		out.write(reinterpret_cast<const char*>(&size), sizeof(uint64_t));
		out.write(reinterpret_cast<const char*>(value.data()), sizeof(type) * size);
		return out;
	}
	friend std::istream& operator >>(std::istream& in, weight& w) {
		std::string name;
    uint32_t len = 0;
    in.read(reinterpret_cast<char *>(&len), sizeof(len));
    name.resize(len);
    in.read(&name[0], len);
		assert(name == w.name());
		// weight
		auto& value = w.value;
		uint64_t size = 0;
		in.read(reinterpret_cast<char*>(&size), sizeof(uint64_t));
		value.resize(size);
		in.read(reinterpret_cast<char*>(value.data()), sizeof(type) * size);
		return in;
	}

protected:
	std::vector<type> value;
	static const size_t iso_level_ = 8;
	std::array<std::vector<score>, iso_level_> isomorphism;
};
