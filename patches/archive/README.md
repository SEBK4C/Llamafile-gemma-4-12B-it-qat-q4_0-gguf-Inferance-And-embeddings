# Archived llama.cpp patch series (provenance)

These patches composed the nested llama.cpp tree from upstream 04eb4c4.
As of v0.7.1 the composed result is a pushed source-true branch
(SEBK4C/llama.cpp@gemma4-v0.7.x, `.gemma4-source-true` marker) — setup
checks out real source instead of reconstructing it, ending the
unpushed-drift class of build breaks. Kept for history and review;
apply-patches.sh skips them on source-true checkouts.
