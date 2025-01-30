#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define MAX_ELEMENTS 100 // Renamed for clarity

// Global variables
int my_array[MAX_ELEMENTS]; // More descriptive name
int array_size; // Meaningful variable name

// Function prototypes
void* parallel_sort(void* args); // More descriptive name
void merge_sorted_halves(int start, int mid, int end); // Clearer action

// Thread arguments struct
struct ThreadParams { // More descriptive name
    int lower_index; // More descriptive variable names
    int upper_index;
};

int main() {
    pthread_t thread1, thread2;
    struct ThreadParams params1, params2;

    // Input size and elements
    printf("Enter the number of elements: ");
    scanf("%d", &array_size);
    printf("Enter %d elements: ", array_size);
    for (int i = 0; i < array_size; i++) {
        scanf("%d", &my_array[i]);
    }

    // Divide array into two halves
    params1.lower_index = 0;
    params1.upper_index = array_size / 2 - 1;
    params2.lower_index = array_size / 2;
    params2.upper_index = array_size - 1;

    // Create threads for sorting
    pthread_create(&thread1, NULL, parallel_sort, &params1);
    pthread_create(&thread2, NULL, parallel_sort, &params2);

    // Wait for threads to finish
    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);

    // Merge the sorted halves
    merge_sorted_halves(0, array_size / 2 - 1, array_size - 1);

    // Display sorted array
    printf("Sorted array: ");
    for (int i = 0; i < array_size; i++) {
        printf("%d ", my_array[i]);
    }
    printf("\n");

    return 0;
}

// Function to sort a portion of the array using bubble sort
void* parallel_sort(void* arguments) {
    struct ThreadParams* params = (struct ThreadParams*)arguments;
    int low = params->lower_index;
    int high = params->upper_index;

    // Bubble sort algorithm
    for (int i = low; i <= high; i++) {
        for (int j = low; j <= high - 1; j++) {
            if (my_array[j] > my_array[j + 1]) {
                int temp = my_array[j];
                my_array[j] = my_array[j + 1];
                my_array[j + 1] = temp;
            }
        }
    }

    // Exit the thread
    pthread_exit(NULL);
}

// Function to merge two sorted halves of the array
void merge_sorted_halves(int start, int mid, int end) {
    int temp_array[MAX_ELEMENTS]; // Consistent naming
    int i = start, j = mid + 1, k = start;

    while (i <= mid && j <= end) {
        if (my_array[i] <= my_array[j]) {
            temp_array[k++] = my_array[i++];
        } else {
            temp_array[k++] = my_array[j++];
        }
    }

    while (i <= mid) {
        temp_array[k++] = my_array[i++];
    }
    while (j <= end) {
        temp_array[k++] = my_array[j++];
    }

    for (int i = start; i <= end; i++) {
        my_array[i]
