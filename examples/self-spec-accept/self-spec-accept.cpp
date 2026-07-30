// self-spec-accept: measure the greedy speculative-decode ACCEPTANCE of a
// "self-draft" — the same model run with a reduced n_expert_used as the draft,
// verified by the same model at full n_expert_used. One copy of the weights is
// shared between the two contexts (mandatory: 2x the model would not fit RAM).
//
// It walks the TARGET's greedy trajectory and, at each position, asks whether the
// DRAFT's top-1 (on the identical prefix) matches. That fraction is exactly the
// per-token acceptance probability of greedy speculative decoding.
//
// Usage:  self-spec-accept -m MODEL -p "prompt" -n 128 [-c 4096]
//         LLAMA_DRAFT_N_EXPERT=2 self-spec-accept ...   (draft active experts; default 2)

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static llama_token argmax(const float * logits, int n) {
    llama_token best = 0;
    float bv = logits[0];
    for (int i = 1; i < n; ++i) {
        if (logits[i] > bv) { bv = logits[i]; best = i; }
    }
    return best;
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_predict = 128;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    params.warmup = false; // skip the all-expert warmup forward (huge for K3); we only measure agreement

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    // comma-separated list of draft active-expert counts to compare in one model load
    std::vector<int> draft_counts;
    {
        const char * env = getenv("LLAMA_DRAFT_N_EXPERT");
        std::string s = env ? env : "2";
        size_t p = 0;
        while (p < s.size()) {
            size_t c = s.find(',', p);
            if (c == std::string::npos) c = s.size();
            int v = atoi(s.substr(p, c - p).c_str());
            if (v > 0) draft_counts.push_back(v);
            p = c + 1;
        }
        if (draft_counts.empty()) draft_counts.push_back(2);
    }

    // target: model + full-expert context (one copy of the weights)
    common_init_result_ptr init = common_init_from_params(params);
    llama_model   * model   = init->model();
    llama_context * ctx_tgt = init->context();
    if (model == nullptr || ctx_tgt == nullptr) {
        fprintf(stderr, "%s: failed to load model / create target context\n", __func__);
        return 1;
    }

    // draft contexts over the SAME model, each with a reduced n_expert_used
    std::vector<llama_context *> ctx_dft;
    for (int nc : draft_counts) {
        llama_context_params cp = common_context_params_to_llama(params);
        cp.n_expert_used = (uint32_t) nc;
        llama_context * c = llama_init_from_model(model, cp);
        if (c == nullptr) {
            fprintf(stderr, "%s: failed to create draft context n_expert_used=%d\n", __func__, nc);
            return 1;
        }
        ctx_dft.push_back(c);
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> inp = common_tokenize(ctx_tgt, params.prompt, true, true);
    if (inp.empty()) {
        fprintf(stderr, "%s: empty prompt\n", __func__);
        return 1;
    }

    fprintf(stderr, "%s: draft n_expert_used sweep = { ", __func__);
    for (int nc : draft_counts) fprintf(stderr, "%d ", nc);
    fprintf(stderr, "} | prompt %zu tok | generating %d\n", inp.size(), params.n_predict);

    // prime target + all draft contexts with the prompt
    if (llama_decode(ctx_tgt, llama_batch_get_one(inp.data(), inp.size())) != 0) {
        fprintf(stderr, "%s: target prompt decode failed\n", __func__);
        return 1;
    }
    for (size_t d = 0; d < ctx_dft.size(); ++d) {
        if (llama_decode(ctx_dft[d], llama_batch_get_one(inp.data(), inp.size())) != 0) {
            fprintf(stderr, "%s: draft[%zu] prompt decode failed\n", __func__, d);
            return 1;
        }
    }

    std::vector<int> n_accept(ctx_dft.size(), 0);
    int n_total = 0;
    std::string out;

    for (int i = 0; i < params.n_predict; ++i) {
        const llama_token tgt = argmax(llama_get_logits_ith(ctx_tgt, -1), n_vocab);
        for (size_t d = 0; d < ctx_dft.size(); ++d) {
            const llama_token dft = argmax(llama_get_logits_ith(ctx_dft[d], -1), n_vocab);
            if (dft == tgt) n_accept[d]++;
        }
        n_total++;

        out += common_token_to_piece(ctx_tgt, tgt);

        if (llama_vocab_is_eog(vocab, tgt)) {
            fprintf(stderr, "%s: EOG at step %d\n", __func__, i);
            break;
        }

        // advance target and all drafts along the TARGET's greedy trajectory
        llama_token tok = tgt;
        bool ok = llama_decode(ctx_tgt, llama_batch_get_one(&tok, 1)) == 0;
        for (size_t d = 0; d < ctx_dft.size() && ok; ++d) {
            ok = llama_decode(ctx_dft[d], llama_batch_get_one(&tok, 1)) == 0;
        }
        if (!ok) { fprintf(stderr, "%s: decode failed at step %d\n", __func__, i); break; }

        // incremental progress so a slow run still shows the number accumulating
        if ((i + 1) % 8 == 0) {
            fprintf(stderr, "%s: [%d/%d]", __func__, i + 1, params.n_predict);
            for (size_t d = 0; d < ctx_dft.size(); ++d) {
                fprintf(stderr, "  e%d=%.0f%%", draft_counts[d], 100.0 * n_accept[d] / n_total);
            }
            fprintf(stderr, "\n");
        }
    }

    // Greedy spec-decode expected accepted tokens per verify pass, chain length K,
    // per-token acceptance p:  E = (1 - p^(K+1)) / (1 - p).
    printf("\n==== self-spec acceptance (target n_expert_used = model default) ====\n");
    printf("tokens compared     : %d\n", n_total);
    printf("%-8s %-10s %-10s  expected target-tokens/verify at K=\n", "draft", "accepted", "rate");
    printf("%-8s %-10s %-10s  %6s %6s %6s %6s\n", "experts", "", "", "2", "4", "6", "8");
    for (size_t d = 0; d < ctx_dft.size(); ++d) {
        const double acc = n_total ? (double) n_accept[d] / n_total : 0.0;
        printf("%-8d %-10d %-9.1f%%", draft_counts[d], n_accept[d], 100.0 * acc);
        for (int K : {2, 4, 6, 8}) {
            double e = (acc >= 1.0) ? (K + 1) : (1.0 - std::pow(acc, K + 1)) / (1.0 - acc);
            printf(" %6.2f", e);
        }
        printf("\n");
    }
    printf("\noutput preview: %.200s\n", out.c_str());

    for (llama_context * c : ctx_dft) llama_free(c);
    // ctx_tgt / model owned by init
    llama_backend_free();
    return 0;
}
