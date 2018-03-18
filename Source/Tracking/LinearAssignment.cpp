#include <vector>
#include <algorithm>
#include "linearassignment.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>     // for seconds()

namespace Tracking {
    namespace LA {
        using namespace std;

        template <typename T>
        struct SSubMin
        {
            SSubMin(T& minvalue): minv(minvalue) {}

            void operator()(T& t) const
            {
                t -= minv;
            }

        private:
            T minv;
        };

        struct SCopyNElements
        {
            SCopyNElements(int num): numElements(num) {};
            template <typename T>
            T operator()(T& t) const
            {
                return T(t.begin(), t.begin()+numElements);
            }

        private:
            int numElements;
        };



        void LinearAssignment(const vector<vector<float> >& costMatrix, float maxGate, vector<vector<int> >& assignmentMatrix)
        {
            size_t numrows = costMatrix.size();
            size_t numcols = costMatrix[0].size();
            // find the min value of costMatrix
            vector<float> temp;
            temp.reserve (numrows);
            for (size_t i = 0; i < numrows; ++i)
            {
                temp.push_back(*min_element(costMatrix[i].begin(), costMatrix[i].end()));
            }
            float minCost = *min_element (temp.begin(), temp.end());
            minCost = min(minCost, maxGate);

            // make sure all element in costMatrix are positive by subtracting the min
            vector<vector<float> > cost(costMatrix);    // intermediate cost matrix
            SSubMin<float> subMin(minCost);
            for (size_t i = 0; i < numrows; ++i)
            {
                for_each(cost[i].begin(), cost[i].end(), subMin);
            }

            float gate = maxGate - minCost;

            // make cost matrix a squared matrix
            if (numrows > numcols)
            {
                for (size_t i = 0; i < numrows; ++i)
                {
                    cost[i].resize (numrows, gate);
                }
            }
            else if (numrows < numcols)
            {
                vector<float> pad;
                pad.resize(numcols, gate);
                cost.resize(numcols, pad);
            }

            // extend cost matrix by doubling its size, pad with the value of gate
            size_t newsize = cost.size();
            vector<float> pad;
            pad.resize(newsize, gate);
            cost.resize(2*newsize, pad);
            for (size_t i = 0; i < cost.size(); ++i)
            {
                cost[i].resize (2*newsize, gate);
            }

            size_t costrows = cost.size();
            size_t costcols = cost[0].size();

            // calculate assignment matrix
            vector<vector<int> > assignment;
            assignment.resize(costrows);
            for (size_t i = 0; i < assignment.size(); ++i)
            {
                assignment[i].resize(costcols);
            }

            LapWrapper(cost, assignment);


            // assign corresponding elements from assignment to assignmentMatrix
            vector<vector<int> >().swap(assignmentMatrix);
            assignmentMatrix.resize(numrows);
            for (size_t i = 0; i < numrows; ++i)
            {
                assignmentMatrix[i].resize(numcols);
            }

            SCopyNElements copyElements(numcols);
            transform(assignment.begin(), assignment.begin()+numrows, assignmentMatrix.begin(), copyElements);
        }



        void LapWrapper(const vector<vector<float> >& costMatrix, vector<vector<int> >& assignMatrix)
        {
            #define COSTRANGE 1000.0
            #define PRINTCOST 0

            int dim, startdim, enddim;
            cost **assigncost, *u, *v, lapcost;
            row i, *colsol;
            col j, *rowsol;
            double runtime;

            //double *xValues;
            int rows, cols;

            //double *outArray;

            cols = static_cast<int>(costMatrix[0].size());
            rows = static_cast<int>(costMatrix.size());

            if(cols != rows) {
                printf("Error: Assignment matrix must be square!\n");
                return;
            }

            enddim = cols;
            startdim = 1;

            assigncost = new cost*[enddim];
            for (i = 0; i < enddim; i++)
                assigncost[i] = new cost[enddim];

            rowsol = new col[enddim];
            colsol = new row[enddim];
            u = new cost[enddim];
            v = new cost[enddim];

            dim = enddim;

            //  for (dim = startdim; dim <= enddim; dim++)
            //  {
            for (j = 0; j < dim; j++) {
                for (i = 0; i < dim; i++) {
                    assigncost[i][j] = (cost) costMatrix[i][j];
                    //	  assigncost[i][j] = (cost) (random_new() * (double) COSTRANGE);
                }
                //	printf("\n");
            }

#if (PRINTCOST) 
            for (i = 0; i < dim; i++)
            {
                printf("\n");
                for (j = 0; j < dim; j++)
                    printf("%4d ", assigncost[i][j]);
            }
#endif


            //    printf("start\n");
            runtime = seconds();
            lapcost = lap(dim, assigncost, rowsol, colsol, u, v);
            //    runtime = seconds() - runtime;
            //    printf("dim  %4d - lap cost %5d - runtime %6.3f\n", dim, lapcost, runtime);
            //checklap(dim, assigncost, rowsol, colsol, u, v);
            //  }

            //plhs[0] = mxCreateDoubleMatrix(enddim, enddim, mxREAL);

            ///* Get a pointer to the data space in our newly allocated memory */
            //outArray = mxGetPr(plhs[0]);
            vector<vector<int> >().swap(assignMatrix);
            assignMatrix.resize(cols);
            for (int i = 0; i < cols; ++i)
            {
                assignMatrix[i].resize(cols);
            }

            for(i = 0; i < enddim; i++) {
                //     printf("Row %d assigned to col %d\n", i, rowsol[i]);
                //outArray[(i*enddim)+rowsol[i]] = 1;
                //       outArray[i] = rowsol[i];
                assignMatrix[i][rowsol[i]] = 1;
            }

            delete[] assigncost;
            delete[] rowsol;
            delete[] colsol;
            delete[] u;
            delete[] v;


        }


        void seedRandom(unsigned int seed)

            // seed for random number generator.

        {

            srand(seed);

            return;   

        }



        double random_new(void)

            // random number between 0.0 and 1.0 (uncluded).

        {

            double rrr;



            rrr = (double) rand() / (double) RAND_MAX;

            return rrr;

        }



        double seconds()

            // cpu time in seconds since start of run.
        {

            double secs;



            secs = (double)(clock() / 1000.0);

            return(secs);

        }


        cost lap(int dim, 
            cost **assigncost,
            col *rowsol, 
            row *colsol, 
            cost *u, 
            cost *v)

            // input:
            // dim        - problem size
            // assigncost - cost matrix

            // output:
            // rowsol     - column assigned to row in solution
            // colsol     - row assigned to column in solution
            // u          - dual variables, row reduction numbers
            // v          - dual variables, column reduction numbers

        {
            boolean unassignedfound;
            row  i, imin, numfree = 0, prvnumfree, f, i0, k, freerow, *pred, *free;
            col  j, j1, j2, endofpath, last, low, up, *collist, *matches;
            cost min, h, umin, usubmin, v2, *d;

            free = new row[dim];       // list of unassigned rows.
            collist = new col[dim];    // list of columns to be scanned in various ways.
            matches = new col[dim];    // counts how many times a row could be assigned.
            d = new cost[dim];         // 'cost-distance' in augmenting path calculation.
            pred = new row[dim];       // row-predecessor of column in augmenting/alternating path.

            // init how many times a row will be assigned in the column reduction.
            for (i = 0; i < dim; i++)  
                matches[i] = 0;

            // COLUMN REDUCTION 
            for (j = dim-1; j >= 0; j--)    // reverse order gives better results.
            {
                // find minimum cost over rows.
                min = assigncost[0][j]; 
                imin = 0;
                for (i = 1; i < dim; i++)  
                    if (assigncost[i][j] < min) 
                    { 
                        min = assigncost[i][j]; 
                        imin = i;
                    }
                    v[j] = min; 

                    if (++matches[imin] == 1) 
                    { 
                        // init assignment if minimum row assigned for first time.
                        rowsol[imin] = j; 
                        colsol[j] = imin; 
                    }
                    else
                        colsol[j] = -1;        // row already assigned, column not assigned.
            }

            // REDUCTION TRANSFER
            for (i = 0; i < dim; i++) 
                if (matches[i] == 0)     // fill list of unassigned 'free' rows.
                    free[numfree++] = i;
                else
                    if (matches[i] == 1)   // transfer reduction from rows that are assigned once.
                    {
                        j1 = rowsol[i]; 
                        min = BIG;
                        for (j = 0; j < dim; j++)  
                            if (j != j1)
                                if (assigncost[i][j] - v[j] < min) 
                                    min = assigncost[i][j] - v[j];
                        v[j1] = v[j1] - min;
                    }

                    // AUGMENTING ROW REDUCTION 
                    int loopcnt = 0;           // do-loop to be done twice.
                    do
                    {
                        loopcnt++;

                        // scan all free rows.
                        // in some cases, a free row may be replaced with another one to be scanned next.
                        k = 0; 
                        prvnumfree = numfree; 
                        numfree = 0;             // start list of rows still free after augmenting row reduction.
                        while (k < prvnumfree)
                        {
                            i = free[k]; 
                            k++;

                            // find minimum and second minimum reduced cost over columns.
                            umin = assigncost[i][0] - v[0]; 
                            j1 = 0; 
                            usubmin = BIG;
                            for (j = 1; j < dim; j++) 
                            {
                                h = assigncost[i][j] - v[j];
                                if (h < usubmin)
                                    if (h >= umin) 
                                    { 
                                        usubmin = h; 
                                        j2 = j;
                                    }
                                    else 
                                    { 
                                        usubmin = umin; 
                                        umin = h; 
                                        j2 = j1; 
                                        j1 = j;
                                    }
                            }

                            i0 = colsol[j1];
                            if (umin < usubmin) 
                                // change the reduction of the minimum column to increase the minimum
                                // reduced cost in the row to the subminimum.
                                v[j1] = v[j1] - (usubmin - umin);
                            else                   // minimum and subminimum equal.
                                if (i0 >= 0)         // minimum column j1 is assigned.
                                { 
                                    // swap columns j1 and j2, as j2 may be unassigned.
                                    j1 = j2; 
                                    i0 = colsol[j2];
                                }

                                // (re-)assign i to j1, possibly de-assigning an i0.
                                rowsol[i] = j1; 
                                colsol[j1] = i;

                                if (i0 >= 0)           // minimum column j1 assigned earlier.
                                    if (umin < usubmin) 
                                        // put in current k, and go back to that k.
                                        // continue augmenting path i - j1 with i0.
                                        free[--k] = i0; 
                                    else 
                                        // no further augmenting reduction possible.
                                        // store i0 in list of free rows for next phase.
                                        free[numfree++] = i0; 
                        }
                    }
                    while (loopcnt < 2);       // repeat once.

                    // AUGMENT SOLUTION for each free row.
                    for (f = 0; f < numfree; f++) 
                    {
                        freerow = free[f];       // start row of augmenting path.

                        // Dijkstra shortest path algorithm.
                        // runs until unassigned column added to shortest path tree.
                        for (j = 0; j < dim; j++)  
                        { 
                            d[j] = assigncost[freerow][j] - v[j]; 
                            pred[j] = freerow;
                            collist[j] = j;        // init column list.
                        }

                        low = 0; // columns in 0..low-1 are ready, now none.
                        up = 0;  // columns in low..up-1 are to be scanned for current minimum, now none.
                        // columns in up..dim-1 are to be considered later to find new minimum, 
                        // at this stage the list simply contains all columns 
                        unassignedfound = FALSE;
                        do
                        {
                            if (up == low)         // no more columns to be scanned for current minimum.
                            {
                                last = low - 1; 

                                // scan columns for up..dim-1 to find all indices for which new minimum occurs.
                                // store these indices between low..up-1 (increasing up). 
                                min = d[collist[up++]]; 
                                for (k = up; k < dim; k++) 
                                {
                                    j = collist[k]; 
                                    h = d[j];
                                    if (h <= min)
                                    {
                                        if (h < min)     // new minimum.
                                        { 
                                            up = low;      // restart list at index low.
                                            min = h;
                                        }
                                        // new index with same minimum, put on undex up, and extend list.
                                        collist[k] = collist[up]; 
                                        collist[up++] = j; 
                                    }
                                }

                                // check if any of the minimum columns happens to be unassigned.
                                // if so, we have an augmenting path right away.
                                for (k = low; k < up; k++) 
                                    if (colsol[collist[k]] < 0) 
                                    {
                                        endofpath = collist[k];
                                        unassignedfound = TRUE;
                                        break;
                                    }
                            }

                            if (!unassignedfound) 
                            {
                                // update 'distances' between freerow and all unscanned columns, via next scanned column.
                                j1 = collist[low]; 
                                low++; 
                                i = colsol[j1]; 
                                h = assigncost[i][j1] - v[j1] - min;

                                for (k = up; k < dim; k++) 
                                {
                                    j = collist[k]; 
                                    v2 = assigncost[i][j] - v[j] - h;
                                    if (v2 < d[j])
                                    {
                                        pred[j] = i;
                                        if (v2 == min)   // new column found at same minimum value
                                            if (colsol[j] < 0) 
                                            {
                                                // if unassigned, shortest augmenting path is complete.
                                                endofpath = j;
                                                unassignedfound = TRUE;
                                                break;
                                            }
                                            // else add to list to be scanned right away.
                                            else 
                                            { 
                                                collist[k] = collist[up]; 
                                                collist[up++] = j; 
                                            }
                                            d[j] = v2;
                                    }
                                }
                            } 
                        }
                        while (!unassignedfound);

                        // update column prices.
                        for (k = 0; k <= last; k++)  
                        { 
                            j1 = collist[k]; 
                            v[j1] = v[j1] + d[j1] - min;
                        }

                        // reset row and column assignments along the alternating path.
                        do
                        {
                            i = pred[endofpath]; 
                            colsol[endofpath] = i; 
                            j1 = endofpath; 
                            endofpath = rowsol[i]; 
                            rowsol[i] = j1;
                        }
                        while (i != freerow);
                    }

                    // calculate optimal cost.
                    cost lapcost = 0;
                    for (i = 0; i < dim; i++)  
                    {
                        j = rowsol[i];
                        u[i] = assigncost[i][j] - v[j];
                        lapcost = lapcost + assigncost[i][j]; 
                    }

                    // free reserved memory.
                    delete[] pred;
                    delete[] free;
                    delete[] collist;
                    delete[] matches;
                    delete[] d;

                    return lapcost;
        }

        void checklap(int dim, cost **assigncost,
            col *rowsol, row *colsol, cost *u, cost *v)
        {
            row  i;
            col  j;
            cost lapcost = 0, redcost = 0;
            boolean *matched;
            char wait;

            matched = new boolean[dim];

            for (i = 0; i < dim; i++)  
                for (j = 0; j < dim; j++)  
                    if ((redcost = assigncost[i][j] - u[i] - v[j]) < 0)
                    {
                        printf("\n");
                        printf("negative reduced cost i %d j %d redcost %d\n", i, j, redcost);
                        printf("\n\ndim %5d - press key\n", dim);
                        scanf_s("%d", &wait);
                        break; 
                    }


            for (i = 0; i < dim; i++)  
                if ((redcost = assigncost[i][rowsol[i]] - u[i] - v[rowsol[i]]) != 0)
                {
                    printf("\n");
                    printf("non-null reduced cost i %d soli %d redcost %d\n", i, rowsol[i], redcost);
                    printf("\n\ndim %5d - press key\n", dim);
                    scanf_s("%d", &wait);
                    break; 
                }

            for (j = 0; j < dim; j++)  
                matched[j] = FALSE;

            for (i = 0; i < dim; i++)  
                if (matched[rowsol[i]])
                {
                    printf("\n");
                    printf("column matched more than once - i %d soli %d\n", i, rowsol[i]);
                    printf("\n\ndim %5d - press key\n", dim);
                    scanf_s("%d", &wait);
                    break; 
                }
                else
                    matched[rowsol[i]] = TRUE;


            for (i = 0; i < dim; i++)  
                if (colsol[rowsol[i]] != i)
                {
                    printf("\n");
                    printf("error in row solution i %d soli %d solsoli %d\n", i, rowsol[i], colsol[rowsol[i]]);
                    printf("\n\ndim %5d - press key\n", dim);
                    scanf_s("%d", &wait);
                    break; 
                }

            for (j = 0; j < dim; j++)  
                if (rowsol[colsol[j]] != j)
                {
                    printf("\n");
                    printf("error in col solution j %d solj %d solsolj %d\n", j, colsol[j], rowsol[colsol[j]]);
                    printf("\n\ndim %5d - press key\n", dim);
                    scanf_s("%d", &wait);
                    break; 
                }


            delete[] matched;
            return;

        }
    }
}