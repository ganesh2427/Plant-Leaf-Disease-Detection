#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define MAX_SIZE 100

// Global variables
int arr[MAX_SIZE];
int size;

// Function prototypes
void* quickSort(void* arg);
void merge(int low, int mid, int high);

// Struct to hold thread arguments
struct ThreadArgs {
    int low;
    int high;
};

int main() {
    pthread_t tid1, tid2;
    struct ThreadArgs args1, args2;

    // Input size and elements
    printf("Enter the number of elements: ");
    scanf("%d", &size);
    printf("Enter %d elements: ", size);
    for (int i = 0; i < size; i++) {
        scanf("%d", &arr[i]);
    }

    // Divide array into two halves
    args1.low = 0;
    args1.high = size / 2 - 1;
    args2.low = size / 2;
    args2.high = size - 1;

    // Create threads for sorting
    pthread_create(&tid1, NULL, quickSort, &args1);
    pthread_create(&tid2, NULL, quickSort, &args2);

    // Wait for threads to finish
    pthread_join(tid1, NULL);
    pthread_join(tid2, NULL);

    // Merge the sorted halves
    merge(0, size / 2 - 1, size - 1);

    // Display sorted array
    printf("Sorted array: ");
    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    return 0;
}

// Function to sort a portion of the array using Quick Sort
void* quickSort(void* arg) {
    struct ThreadArgs* args = (struct ThreadArgs*)arg;
    int low = args->low;
    int high = args->high;

    if (low < high) {
        int pivot = arr[high];
        int i = low - 1;

        for (int j = low; j <= high - 1; j++) {
            if (arr[j] <= pivot) {
                i++;
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
        int temp = arr[i + 1];
        arr[i + 1] = arr[high];
        arr[high] = temp;

        int partitionIndex = i + 1;

        struct ThreadArgs args_left = {low, partitionIndex - 1};
        struct ThreadArgs args_right = {partitionIndex + 1, high};

        pthread_t tid_left, tid_right;
        pthread_create(&tid_left, NULL, quickSort, &args_left);
        pthread_create(&tid_right, NULL, quickSort, &args_right);

        pthread_join(tid_left, NULL);
        pthread_join(tid_right, NULL);
    }

    // Exit the thread
    pthread_exit(NULL);
}

// Function to merge two sorted halves of the array
void merge(int low, int mid, int high) {
    int merged[MAX_SIZE];
    int i = low, j = mid + 1, k = 0;

    while (i <= mid && j <= high) {
        if (arr[i] <= arr[j]) {
            merged[k++] = arr[i++];
        } else {
            merged[k++] = arr[j++];
        }
    }

    while (i <= mid) {
        merged[k++] = arr[i++];
    }
    while (j <= high) {
        merged[k++] = arr[j++];
    }

    // Copy elements back to original array
    for (int i = low, k = 0; i <= high; i++, k++) {
        arr[i] = merged[k];
    }
}
