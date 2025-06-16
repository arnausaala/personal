#include <stdlib.h>

int main(int argc, char *argv[]) {

    //...
    struct node *p = init_list();

    #pragma omp parallel
    {
        #pragma omp single
        {

            while (p != NULL) {
                #pragma omp task firstprivate(p)
                {
                    processwork(p);
                }
                p = p->next;
            }
        }
    }
    
    //...
    return 0;
}

struct node {
    float data;
    struct node* next;
};