#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define MAX_SIZE 100

// Global variables
int array[MAX_SIZE];
int size;

// Function prototypes
void* sort(void* arg);
void merge(int low, int mid, int high);

// Struct for thread arguments
struct ThreadData {
    int low;
    int high;
};

int main() {
    pthread_t thread1, thread2;
    struct ThreadData data1, data2;

    // Input size and elements
    printf("Enter the number of elements: ");
    scanf("%d", &size);
    printf("Enter %d elements: ", size);
    for (int i = 0; i < size; i++) {
        scanf("%d", &array[i]);
    }

    // Divide array into two halves
    data1.low = 0;
    data1.high = size / 2 - 1;
    data2.low = size / 2;
    data2.high = size - 1;

    // Create threads for sorting
    pthread_create(&thread1, NULL, sort, &data1);
    pthread_create(&thread2, NULL, sort, &data2);

    // Wait for threads to finish
    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);

    // Merge the sorted halves
    merge(0, size / 2 - 1, size - 1);

    // Display sorted array
    printf("Sorted array: ");
    for (int i = 0; i < size; i++) {
        printf("%d ", array[i]);
    }
    printf("\n");

    return 0;
}

// Function to sort a portion of the array
void* sort(void* arg) {
    struct ThreadData* data = (struct ThreadData*)arg;
    int low = data->low;
    int high = data->high;

    // Bubble sort algorithm
    for (int i = low; i <= high; i++) {
        for (int j = low; j <= high - 1; j++) {
            if (array[j] > array[j + 1]) {
                int temp = array[j];
                array[j] = array[j + 1];
                array[j + 1] = temp;
            }
        }
    }
    
    // Exit the thread
    pthread_exit(NULL);
}

// Function to merge two sorted halves of the array
void merge(int low, int mid, int high) {
    int merged[MAX_SIZE];
    int i = low, j = mid + 1, k = low;

    while (i <= mid && j <= high) {
        if (array[i] <= array[j]) {
            merged[k++] = array[i++];
        } else {
            merged[k++] = array[j++];
        }
    }

    while (i <= mid) {
        merged[k++] = array[i++];
    }
    while (j <= high) {
        merged[k++] = array[j++];
    }

    for (int i = low; i <= high; i++) {
        array[i] = merged[i];
    }
}
