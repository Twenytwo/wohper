
---

## 2026-07-09 - TOOL CALLING DSML nello shim: agenti (OpenClaw) pilotano DeepSeek locale

Costruito il pezzo che mancava per usare Wohper come cervello di agenti.
SCOPERTA CHIAVE: il modello ha token tool dedicati (128806-128814,
<｜DSML｜...) e - jackpot - DeepSeek spedisce COL MODELLO (models/.../
encoding/encoding_dsv4.py, Python puro) l'encoder+parser ufficiali che
rendono i tools OpenAI in DSML e ri-parsano. Import dalla dir del modello
(arriva col download HF), niente vendoring/licenza.

Shim riscritto: build_prompt_ids usa encoding_dsv4.encode_messages
(thinking_mode chat/thinking, tools sul system message); parse_completion
usa un PARSER DSML TOLLERANTE (regex, keyed sui tag di apertura +
parametri) - NON il parser ufficiale che è troppo severo (raise su
chiusure imperfette: il modello a freddo ha emesso `</｜DSML｜inv>` invece
di `invoke>`). Il tollerante recupera comunque la call.

GATE E2E: richiesta con tools get_weather → risposta OpenAI perfetta
(content:null, tool_calls[0].function=get_weather({"city":"Rome"}),
finish_reason:tool_calls, id+index aggiunti). Chat semplice INVARIATA
(regression: capital-of-France 7 token). Round-trip risultato tool
(role:tool → <tool_result>) gestito dall'encoder ufficiale → loop
multi-step chiuso. Tool req sono bufferizzate (blocco DSML parsabile solo
completo); se stream richiesto → burst SSE unico; plain chat streamma live.

docs/agent-integration.md: setup OpenClaw (provider custom baseUrl
:8114/v1) + nota velocità onesta (async/background, non real-time).
OpenClaw issue #85918 confermava il gap DSML - ora risolto lato nostro.
Push su main. APERTO: streaming tool-call incrementale vero (ora burst);
test e2e con OpenClaw installato (lato utente).
