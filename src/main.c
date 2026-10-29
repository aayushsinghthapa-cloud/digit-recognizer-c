/* main.c - command-line entry point: dispatches ./digits <command>. */
#include <stdio.h>
#include <string.h>

/* Print the list of available commands. */
static void usage(void)
{
    printf("usage: ./digits <command>\n"
           "commands are added stage by stage (show, knn, train-softmax, train-nn, ...)\n");
}

int main(int argc, char **argv)
{
    if (argc < 2 || strcmp(argv[1], "help") == 0) {
        usage();
        return 0;
    }
    fprintf(stderr, "unknown command: %s\n", argv[1]);
    usage();
    return 1;
}
