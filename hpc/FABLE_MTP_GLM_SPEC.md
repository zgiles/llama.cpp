# MTP/nextn speculative decode for GLM_DSA (GLM-5.2) — implementation spec (Fable, 2026-07-16)

## Phase 0 pre-gate: PASSED (ran for real)
REAP GLM-5.2 nextn (blk.78.nextn.*) weights are HEALTHY and BIT-IDENTICAL to the full non-REAP UD-Q6's
nextn -> REAP did not touch the MTP head. eh_proj mean +0.00012 std 0.0199, no dead rows. Expect
alpha ~0.5-0.7 (only the MTP block's routed experts were REAP-pruned). MTP block = blk.78 (n_layer=78,
nextn_predict_layers=1), all in shard 8, +5.05 GiB RAM to enable. nextn.embed_tokens + shared_head_head
ABSENT -> fall back to model.tok_embd / model.output. Repro: gguf-py dequantize blk.78.nextn.* from
GLM-5.2-REAP-...00008-of-00008.gguf.

## CRITICAL: glm-dsa.cpp:81 sets TENSOR_SKIP for i>=n_layer -> the whole blk.78 is never allocated/loaded
(create_tensor returns nullptr). The loader change (1a) is mandatory. NEXTN_* tensor-info ops in
llama-arch.cpp:786-791 are real (MUL_MAT/MUL/GET_ROWS) so dropping TENSOR_SKIP loads them fine (qwen35moe/
step35 already do).

## Phase 1 (single socket)
### 1a Loader (glm-dsa.cpp load_arch_tensors): actually load blk.78.
Probe like step35: has_mtp = n_layer_nextn>0 && ml.get_weight("blk.<n_layer>.nextn.eh_proj.weight")!=null.
For i>=n_layer: if has_mtp -> flags |= TENSOR_NOT_REQUIRED (load it); else old TENSOR_SKIP|NOT_REQUIRED.
embed_tokens/shared_head_head stay NOT_REQUIRED (absent). +5.05 GiB.

### 1b deepseek2.cpp graph tail (SHARED w/ mistral4/ocr - gate carefully): set res->t_h_nextn.
h_nextn_full = cparams.embeddings_nextn && !cparams.embeddings_nextn_masked.
At il==n_layer-1: keep the existing inp_out_ids row-trim EXCEPT when h_nextn_full.
After `cur = build_norm(cur, model.output_norm,...)`: cb(cur,"h_nextn",-1); res->t_h_nextn = cur;
if (h_nextn_full && inp_out_ids) cur = ggml_get_rows(ctx0, cur, inp_out_ids); cb "result_norm";
res->t_embd = cur. (MTP seed = POST-output_norm hidden, per qwen35moe precedent. ctx_tgt created with
llama_set_embeddings_nextn(true, masked=false) so needs all rows. can_reuse compares embeddings_nextn/
_masked - safe.)

### 1c graph_mtp (new, glm-dsa.cpp + models.h). models.h: add `struct graph_mtp : llm_graph_context {
graph_mtp(const llama_model&, const llm_graph_params&);};` in llama_model_glm_dsa. build_arch_graph:
dispatch on params.gtype==LLM_GRAPH_TYPE_DECODER_MTP like qwen35moe.cpp:153-158.
Constructor (il=78, layer=model.layers[il]) = qwen35moe MTP scaffolding + deepseek2 MLA + deepseek2 MoE:
 1. asserts eh_proj/enorm/hnorm + wq_a/wq_b/wkv_a_mqa/wk_b/wv_b/ffn_gate_inp non-null.
 2. dims + YaRN kq_scale: copy deepseek2.cpp:151-172 VERBATIM (head_k_mla 256, rope 64, nope 192,
    kv_lora 512, mscale/kq_scale prescale). NOT qwen35moe's 1/sqrt(head). skip inp_attn_scale.
 3. inputs: copy qwen35moe.cpp:588-618 (llm_graph_input_embd_h(n_embd), tokens/embd/h; tok_embd =
    get_rows(layer.nextn.embed_tokens?:model.tok_embd, inp->tokens); add_input; build_inp_pos;
    build_inp_out_ids) BUT `auto* inp_attn = build_attn_inp_k();` (MLA K-only), NOT build_attn_inp_kv().
 4. seed: h_norm=build_norm(inp->h, hnorm, RMS); e_norm=build_norm(tok_embd, enorm, RMS);
    concat=ggml_concat(e_norm,h_norm,0); cur=build_lora_mm(nextn.eh_proj, concat, eh_proj_s); inpSA=cur.
 5. MLA attn: build_norm(cur, layer.attn_norm, RMS, il); copy deepseek2.cpp:234-328 non-lite MLA branch
    (wq_a->q_a_norm->wq_b; q_nope/q_pe; wkv_a_mqa->kv_cmpr/k_pe; rope_ext on q_pe/k_pe; kv_a_norm; wk_b
    absorb; Qcur=concat(q_nope_absorbed,q_pe), Kcur=concat(kv_cmpr,k_pe), Vcur=kv_cmpr); then
    cur = build_attn(inp_attn, layer.wo, NULL, layer.wo_s, Qcur,Kcur,Vcur, nullptr,nullptr, layer.wv_b,
    kq_scale, il);  <- PASS wo -> internal TP allreduce. NO manual allreduce.
 6. residual+MoE: cur=add(cur,inpSA); ffn_inp=cur; build_norm(cur,layer.ffn_norm); copy deepseek2.cpp:
    387-414 (build_moe_ffn w/ ffn_gate_inp, up/gate/down_exps, exp_probs_b, n_expert, n_expert_used,
    SILU, expert_weights_norm, expert_weights_scale, (gating_func)expert_gating_func, il, nullptr,
    ffn_gate_up_exps) + shexp build_ffn(...SILU,PAR,il) + add; cur=add(cur,ffn_inp). SIGMOID gating (hparams),
    not hardcoded SOFTMAX.
 7. head: head_norm_w = nextn.shared_head_norm?:model.output_norm; build_norm(-1); cb"h_nextn";
    res->t_h_nextn=cur; get_rows(cur,inp_out_ids); head_w = nextn.shared_head_head?:model.output;
    build_lora_mm->res->t_logits; build_forward_expand. (skip DSA indexer, cvec, temp scaling.)

### 1d KV filter (llama-model.cpp:2148): extend the step35 branch to GLM_DSA.
if ((arch==STEP35||arch==GLM_DSA) && n_layer_nextn>0) { if ctx_type==MTP: filter = il>=n_layer();
else filter = il<n_layer(); }  (MLA KV uniform; also stops main ctx wasting blk.78's KV.)

### 1e spec wiring: NOTHING to add. No arch whitelist for draft-mtp. Gates pass: MTP ctx needs
n_layer_nextn>0 (glm-dsa=1); draft-mtp path generic (n_mtp_layers=1 single-head); rollback = plain
seq_rm (SEQ_RM_TYPE_PART, n_rs_seq=0, NOT the RS path). llama-cli embeds server_context.

## Phase 2 TP: near-zero. Head-div already n_layer_all (commit ab9508367). Weights shard by llm_tensor
enum not layer (blk.78 attn_q_b COLUMN, k_b/v_b EXPERT-slice, attn_output ROW, ffn_*_exps per moe_mode
168%2/4=0, NEXTN_* -> TP_SHARD_NONE replicated). DO NOT add manual allreduce (wo passed to build_attn ->
internal reduce; build_moe_ffn self-reduces; shexp name-guard suppressed). Run first with LLAMA_TP_MOE=
expert (prior glm52 config), then optionally +LLAMA_TP_ATTN.

## Verify (Phase 1, single socket; build w/ -DLLAMA_BUILD_SERVER=ON + openssl; -st + timeout ALWAYS):
control: numactl -N0 -m0 llama-cli -m REAP-shard1 -t 16 --temp 0 -n 160 -st -f prompt2
mtp:     + --spec-type draft-mtp --draft-max 4 --draft-min 1
Gates: (1) log "adding speculative implementation 'draft-mtp'" + nonzero accepted, no GGML_ASSERT.
(2) alpha from draft stats or t/s via rho(K); alpha<0.1 -> stop, try full UD-Q6. (3) exactness =
determinism (2 spec runs identical) + spec-vs-control benign near-tie divergence. (4) K sweep 2/4/6/8
(qwen peaked 6). Phase 2 = 2-rank per-socket numactl, UCX_TLS=sm,self, LLAMA_TP_SIZE=2, LLAMA_TP_MOE=
expert, --spec-type draft-mtp; pass = lockstep + coherent.

## Opus checklist: nextn *_s scale companions exist (null for glm-dsa, pass thru build_lora_mm);
model.output non-null (tok_embd-dup glm-dsa.cpp:71-74); cb tags prefix mtp_; don't touch rope_yarn_log_mul.
After 1a: +5.05GiB RAM, "unused tensor blk.78.*" warnings stop.
