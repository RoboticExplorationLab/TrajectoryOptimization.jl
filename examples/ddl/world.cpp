#include "world.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

#include "mpc_functions.h"
#include "utilities.h"

static double norm(double E, double N);
static double crossSign(double u1, double u2, double v1, double v2);

World::World(){};

World::World(std::string worldName){
  std::cout << "World: opening " << worldName << std::endl;

  // roslaunch executes owl from ~/.ros
  std::ifstream worldFile(worldName);

  if ( !worldFile.is_open() ) {
    std::cerr << "ERROR: couldn't find " << worldName << std::endl;
    return;
  }

  // from python:
  //fieldList = ['s_m', 'posE_m', 'posN_m', 'psi_rad', 'k_1pm', 'grade_rad', 'edgeL_m', 'edgeR_m', 'UxDes_mps']

  std::cout << "World: reading field ";

  std::string line;
  std::getline(worldFile, line);
  parseLine(line, this->s_m);
  std::getline(worldFile, line);
  parseLine(line, this->posE_m);
  std::getline(worldFile, line);
  parseLine(line, this->posN_m);
  std::getline(worldFile, line);
  parseLine(line, this->psi_rad);
  std::getline(worldFile, line);
  parseLine(line, this->k_1pm);
  std::getline(worldFile, line);
  parseLine(line, this->grade_rad);
  std::getline(worldFile, line);
  parseLine(line, this->edgeL_m);
  std::getline(worldFile, line);
  parseLine(line, this->edgeR_m);
  std::getline(worldFile, line);
  parseLine(line, this->UxDes_mps);

  // check that all fields are same length
  if ( s_m.size()!=posE_m.size() || s_m.size()!=posN_m.size() || s_m.size()!=psi_rad.size() ||
      s_m.size()!=k_1pm.size() || s_m.size()!=grade_rad.size() || s_m.size()!=edgeL_m.size() ||
      s_m.size()!=edgeR_m.size() || s_m.size()!=UxDes_mps.size() ) {
    std::cout << "WARNING: world fields must be the same length" << std::endl;
  }

  std::cout << std::endl << "World: complete" << std::endl;

  /*
     for(auto d : this->posE_m){
     std::cout << d << std::endl;
     }
     */

}

void World::parseLine(std::string line, std::vector<double>& field){
  std::stringstream sline(line);
  std::string fieldname;
  std::string token;
  double element;

  sline >> fieldname;
  std::cout << fieldname << ", ";

  while ( getline(sline,token, ',' ) ) {
    element = atof(token.c_str());
    field.push_back(element);
  }
}

/*  Convert local path [distance s, lateral deviation e] to global [East,North] coordinates. */
void World::convertPathToGlobal(double interpS, double e, double& E, double& N) const {

  double nomPosE;
  double nomPosN;
  double nomPsi;

  // check if map is closed
  if( interpS>s_m.back() && !isOpen ) {
    interpS = interpS - s_m.back();
  }

  // store interpolated value in E and interpIdx
  InterpData_T interp = interpolate1D(s_m, posE_m, s_m.size(), interpS, nomPosE);
  const_time_interp(interp, posN_m, nomPosN);
  const_time_interp(interp, psi_rad, nomPsi);

  // account for lateral error from path
  E = nomPosE - e*cos(nomPsi);
  N = nomPosN - e*sin(nomPsi);
}

void World::convertGlobalToPath(double sSeed, double E, double N, double& s, double& e) const {
  /* look for closest two adjacent points (pair) to the E,N position around the seed values locally */
  int sStartIdx;
  if (sSeed < s_m[0]) {
    sStartIdx = 0;
  } else if (sSeed > s_m[s_m.size()-1]) {
    sStartIdx = s_m.size()-1;
  } else {
    sStartIdx = binarySearch(s_m, s_m.size(), sSeed)+1;
  }

  // forward search
  double lastPair = HUGE_VAL; // infinity
  unsigned int forwardIdx = sStartIdx;
  bool stillDecreasing = true;
  while (stillDecreasing) {
    double currentPair;
    if (forwardIdx+1 < s_m.size()) {
      currentPair = norm(E-posE_m[forwardIdx],   N-posN_m[forwardIdx]) +
        norm(E-posE_m[forwardIdx+1], N-posN_m[forwardIdx+1]);
    } else {
      // should only be here if sStartIdx was the last index
      currentPair = HUGE_VAL;
    }

    stillDecreasing = (currentPair < lastPair);
    if (stillDecreasing) {
      lastPair = currentPair;
      forwardIdx++;
    }
  }
  double smallestF = lastPair;

  // backward search
  lastPair = HUGE_VAL; // infinity
  int backwardIdx = sStartIdx;
  stillDecreasing = true;
  while (stillDecreasing) {
    double currentPair;
    if (backwardIdx-1 > 0) {
      currentPair = norm(E-posE_m[backwardIdx],   N-posN_m[backwardIdx]) +
        norm(E-posE_m[backwardIdx-1], N-posN_m[backwardIdx-1]);
    } else {
      // should only be here if sStartIdx was the 0th index
      currentPair = HUGE_VAL;
    }

    stillDecreasing = (currentPair < lastPair);
    if (stillDecreasing) {
      lastPair = currentPair;
      backwardIdx--;
    }
  }
  double smallestB = lastPair;

  int loSIdx;
  int hiSIdx;
  if (smallestB < smallestF) {
    loSIdx = backwardIdx;
    hiSIdx = backwardIdx+1;
  } else {
    loSIdx = forwardIdx-1;
    hiSIdx = forwardIdx;
  }

  // calculating the distances between currrent (E,N), path (E,N) at lo index, and path (E,N) at hi index
  double a = norm( E-posE_m[loSIdx], N-posN_m[loSIdx]);
  double b = norm( E-posE_m[hiSIdx], N-posN_m[hiSIdx]);
  double c = norm( posE_m[loSIdx]-posE_m[hiSIdx], posN_m[loSIdx]-posN_m[hiSIdx]);

  // distance from path at loSIdx to current (E,N) projected onto the path
  double deltaS = ( pow(a,2)+pow(c,2)-pow(b,2) )/(2.0*c);

  s = s_m[loSIdx] + deltaS;
  double absE = sqrt(std::abs(pow(a,2) - pow(deltaS,2)));

  /* to determine sign of e, cross the heading vector at the minimum distance point from the vehicle position with
   * the vector from the min distance point to the vehicle position. The sign of the result is the sign of e */
  double signE = crossSign( posE_m[hiSIdx]-posE_m[loSIdx], posN_m[hiSIdx]-posN_m[loSIdx], E-posE_m[loSIdx], N-posN_m[loSIdx]);
  e = absE*signE;

}

/* Mapmatching without seeding
   finds nearest (posE, posN) point on road and uses that as seed */
void World::convertGlobalToPath(double E, double N, double& s, double& e) const {
  double smallestSoFar = norm(E-posE_m[0], N-posE_m[0]);
  int smallestSoFar_idx = 0;

  for (unsigned int i=1; i<s_m.size(); ++i) {
    double thisDistance = norm(E-posE_m[i], N-posN_m[i]);
    if (thisDistance < smallestSoFar) {
      smallestSoFar = thisDistance;
      smallestSoFar_idx = i;
    }
  }
  convertGlobalToPath(s_m[smallestSoFar_idx], E, N, s, e);
}

/* calculates the Euclidean norm (2-norm) for position in E,N */
static double norm(double E, double N) {
  return sqrt(E*E + N*N);
}

static double crossSign(double u1, double u2, double v1, double v2) {
  double cross = u1*v2 - v1*u2;
  if (cross > 0) {
    return 1.0;
  } else if (cross < 0) {
    return -1.0;
  }
  return 0.0;
}
