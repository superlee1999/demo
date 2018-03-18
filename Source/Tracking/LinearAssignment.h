#ifndef TRACKING_LINEARASSIGNMENT_H
#define TRACKING_LINEARASSIGNMENT_H
#include <vector>

// original code adapted from colleague in University of Bristol

namespace Tracking {
    namespace LA {
        /*************** CONSTANTS  *******************/
#define BIG 100000
        /*************** TYPES      *******************/
        typedef int row;
        typedef int col;
        typedef float cost;
        /*************** FUNCTIONS  *******************/
#if !defined TRUE
#define	 TRUE		1
#endif
#if !defined FALSE
#define  FALSE		0
#endif
        /*************** DATA TYPES *******************/
        typedef int boolean;
        /************************/

        void LinearAssignment(const std::vector<std::vector<float> >& costMatrix, float maxGate, std::vector<std::vector<int> >& assignmentMatrix);
        void LapWrapper(const std::vector<std::vector<float> >& costMatrix, std::vector<std::vector<int> >& assignMatrix);
        cost lap(int dim, cost **assigncost, int *rowsol, int *colsol, cost *u, cost *v);
        void checklap(int dim, cost **assigncost, int *rowsol, int *colsol, cost *u, cost *v);
        void seedRandom(unsigned int seed);
        double random_new(void);
        double seconds();

    }
}


#endif