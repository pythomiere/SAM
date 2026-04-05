#include "model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <ctype.h>
#include <sys/wait.h>
#include <errno.h>
#include <stdint.h>

#define MAX_SENTENCE 1024
#define MAX_TOKENS 256
#define NUM_WORKERS 5

typedef struct {
    uint32_t job_id;
    uint32_t len;
    char text[MAX_SENTENCE];
} SentenceMsg;

typedef struct {
    uint32_t job_id;
    float score;
} Result;

static ssize_t read_full(int fd, void *buf, size_t n) {
  size_t done = 0;
  while (done < n) {
    ssize_t r = read(fd, (char*)buf + done, n - done);
    if (r < 0) {
      if (errno == EINTR) continue;
      return -1;
    }
    if (r == 0) return 0;
    done += r;
  }
  return (ssize_t)done;
}

static void worker(int read_fd, int write_fd, Weights *w, Vocab *v) {
    SentenceMsg msg;
    while (1) {
        ssize_t got = read_full(read_fd, &msg, sizeof(msg));
        if (got <= 0 || msg.len == 0)
            break;

        int indices[MAX_TOKENS];
        int count = tokenize(msg.text, v, indices, MAX_TOKENS);
        float score = predict(indices, count, w);

        Result res;
        res.job_id = msg.job_id;
        res.score = score;

        if (write(write_fd, &res, sizeof(res)) != (ssize_t)sizeof(res)) {
            perror("worker write");
            break;
        }
    }

    close(read_fd);
    close(write_fd);
    exit(0);
}

int main(int argc, char **argv) {
    int show_all = 0;
    int opt;
    while ((opt = getopt(argc, argv, "a")) != -1) {
        if (opt == 'a')
            show_all = 1;
        else {
            fprintf(stderr, "usage: %s [-a]\n", argv[0]);
            return 1;
        }
    }

    Weights weights;
    Vocab vocab;

    if (load_weights("weights.bin", &weights) != 0) {
        fprintf(stderr, "failed to load weights\n");
        return 1;
    }
    if (load_vocab("vocab.bin", &vocab) != 0) {
        fprintf(stderr, "failed to load vocab\n");
        return 1;
    }

    int to_worker[NUM_WORKERS][2];
    int from_worker[NUM_WORKERS][2];
    pid_t pids[NUM_WORKERS];

    for (int w = 0; w < NUM_WORKERS; w++) {
        if (pipe(to_worker[w]) != 0 || pipe(from_worker[w]) != 0) {
            perror("pipe");
            return 1;
        }
    }

    for (int w = 0; w < NUM_WORKERS; w++) {
        pid_t pid = fork();
        if (pid < 0) {
            perror("fork");
            return 1;
        }

        if (pid == 0) {
            for (int i = 0; i < NUM_WORKERS; i++) {
                if (i == w) {
                    close(to_worker[i][1]);
                    close(from_worker[i][0]);
                } else {
                    close(to_worker[i][0]);
                    close(to_worker[i][1]);
                    close(from_worker[i][0]);
                    close(from_worker[i][1]);
                }
            }
            worker(to_worker[w][0], from_worker[w][1], &weights, &vocab);
        }

        close(to_worker[w][0]);
        close(from_worker[w][1]);
        pids[w] = pid;
    }

    fprintf(stderr, "Started %d workers. Reading from stdin...\n\n", NUM_WORKERS);

    size_t buf_cap = 4096, buf_len = 0;
    char *buf = malloc(buf_cap);
    if (!buf) { perror("malloc"); return 1; }
    {
        int c;
        while ((c = fgetc(stdin)) != EOF) {
            if (buf_len + 1 >= buf_cap) {
                buf_cap *= 2;
                buf = realloc(buf, buf_cap);
                if (!buf) { perror("realloc"); return 1; }
            }
            buf[buf_len++] = (char)c;
        }
        buf[buf_len] = '\0';
    }

    uint32_t capacity = 64;
    float *scores = malloc(capacity * sizeof(float));
    char **lines = malloc(capacity * sizeof(char *));
    if (!scores || !lines) { perror("malloc"); return 1; }

    uint32_t total = 0;
    int next_worker = 0;
    int collect_idx = 0;
    int active_jobs = 0;

    char *p = buf;
    while (*p) {
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;

        char *start = p;
        while (*p && *p != '.' && *p != '!' && *p != '?' && *p != '\n') p++;
        if (*p) p++;

        size_t len = (size_t)(p - start);
        while (len > 0 && isspace((unsigned char)start[len - 1])) len--;
        if (len == 0) continue;
        if (len >= MAX_SENTENCE) len = MAX_SENTENCE - 1;

        char sentence[MAX_SENTENCE];
        memcpy(sentence, start, len);
        sentence[len] = '\0';

        if (active_jobs == NUM_WORKERS) {
            Result res;
            ssize_t got = read_full(from_worker[collect_idx][0], &res, sizeof(res));
            if (got == (ssize_t)sizeof(res)) {
                scores[res.job_id] = res.score;
            } else {
              fprintf(stderr, "failed to read result from worker %d\n", collect_idx);
            }
            active_jobs--;
            collect_idx = (collect_idx + 1) % NUM_WORKERS;
        }

        if (total >= capacity) {
            capacity *= 2;
            scores = realloc(scores, capacity * sizeof(float));
            lines = realloc(lines, capacity * sizeof(char *));
            if (!scores || !lines) { perror("realloc"); return 1; }
        }

        lines[total] = strdup(sentence);
        if (!lines[total]) { perror("strdup"); return 1; }

        SentenceMsg msg;
        msg.job_id = total;
        msg.len = (uint32_t)len;
        memcpy(msg.text, sentence, len + 1);

        if (write(to_worker[next_worker][1], &msg, sizeof(msg)) != (ssize_t)sizeof(msg)) {
            perror("write to worker");
            return 1;
        }

        active_jobs++;
        total++;
        next_worker = (next_worker + 1) % NUM_WORKERS;
    }
    free(buf);

    while (active_jobs > 0) {
        Result res;
        ssize_t got = read_full(from_worker[collect_idx][0], &res, sizeof(res));
        if (got == (ssize_t)sizeof(res)) {
            scores[res.job_id] = res.score;
        } else {
          fprintf(stderr, "error: failed to read result from worker %d\n", collect_idx);
        }
        active_jobs--;
        collect_idx = (collect_idx + 1) % NUM_WORKERS;
    }

    SentenceMsg shutdown_msg;
    memset(&shutdown_msg, 0, sizeof(shutdown_msg));
    for (int i = 0; i < NUM_WORKERS; i++) {
        if (write(to_worker[i][1], &shutdown_msg, sizeof(shutdown_msg)) < 0)
            perror("write shutdown");
        close(to_worker[i][1]);
    }

    for (int w = 0; w < NUM_WORKERS; w++)
        close(from_worker[w][0]);

    for (int w = 0; w < NUM_WORKERS; w++) {
        int status;
        waitpid(pids[w], &status, 0);
    }

    if (show_all) {
        printf("\n%-55s %-8s %s\n", "Sentence", "Label", "Score");
        printf("%-55s %-8s %s\n",
               "-------------------------------------------------------",
               "--------", "-------");
        for (uint32_t i = 0; i < total; i++) {
            const char *label = scores[i] > 0.5f ? "Positive" : "Negative";
            char display[56];
            strncpy(display, lines[i], 55);
            display[55] = '\0';
            printf("%-55s %-8s %.4f\n", display, label, scores[i]);
        }
    } else {
        uint32_t pos = 0;
        float sum = 0.0f;
        for (uint32_t i = 0; i < total; i++) {
            sum += scores[i];
            if (scores[i] > 0.5f) pos++;
        }
        uint32_t neg = total - pos;
        float avg = total > 0 ? sum / (float)total : 0.0f;
        const char *overall = avg > 0.5f ? "Positive" : "Negative";

        printf("\nSentences analyzed : %u\n", total);
        printf("Positive           : %u (%.1f%%)\n", pos,
               total > 0 ? 100.0f * pos / (float)total : 0.0f);
        printf("Negative           : %u (%.1f%%)\n", neg,
               total > 0 ? 100.0f * neg / (float)total : 0.0f);
        printf("Average score      : %.4f\n", avg);
        printf("Overall sentiment  : %s\n", overall);
    }

    for (uint32_t i = 0; i < total; i++)
        free(lines[i]);

    free(scores);
    free(lines);
    free_weights(&weights);
    free_vocab(&vocab);
    return 0;
}
