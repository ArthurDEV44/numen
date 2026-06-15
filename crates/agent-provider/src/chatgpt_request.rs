//! Construction du corps de requête Responses API (backend ChatGPT/Codex) depuis
//! le `CanonicalRequest` (Anthropic-like, transcript client-side). Transcrit
//! verbatim du wire format Pi (`openai-codex-responses.ts` +
//! `openai-responses-shared.ts`, vérifié contre le code).
//!
//! Invariants load-bearing :
//! - `store: false` TOUJOURS (le backend rejette `true`).
//! - system prompt → `instructions` (string), JAMAIS un item `input[]`.
//! - SSE **stateless** : pas de `previous_response_id` → contexte complet dans
//!   `input[]` à chaque tour (mappe le canonique, ARCHITECTURE/PROVIDERS §4.1).
//! - `call_id` corrèle `function_call` ↔ `function_call_output`.
//!
//! ⚠️ Reasoning items NON réinjectés en MVP (voir `chatgpt.rs` — risque #1 à
//! valider en live ; le modèle raisonne et on affiche le résumé, mais on ne
//! renvoie pas les items chiffrés au tour suivant).

use agent_core::message::{ContentBlock, Message, Role};
use agent_core::provider::{CanonicalRequest, ToolSpec};
use serde_json::{Value, json};

const DEFAULT_INSTRUCTIONS: &str = "You are a helpful assistant.";

/// Construit le corps JSON complet de la requête Responses (SSE).
pub fn build_responses_body(req: &CanonicalRequest, reasoning_effort: Option<&str>) -> Value {
    let instructions = req.system.as_deref().unwrap_or(DEFAULT_INSTRUCTIONS);

    let mut body = json!({
        "model": req.model,
        // load-bearing : le backend Codex rejette store:true.
        "store": false,
        "stream": true,
        "instructions": instructions,
        "input": build_input(&req.messages),
        "text": { "verbosity": "low" },
        "include": ["reasoning.encrypted_content"],
        "tool_choice": "auto",
        "parallel_tool_calls": true,
    });

    if !req.tools.is_empty() {
        body["tools"] = build_tools(&req.tools);
    }
    if let Some(effort) = reasoning_effort {
        body["reasoning"] = json!({ "effort": effort, "summary": "auto" });
    }
    body
}

/// Convertit le transcript canonique en `input[]` de la Responses API.
fn build_input(messages: &[Message]) -> Value {
    let mut input: Vec<Value> = Vec::new();
    for msg in messages {
        match msg.role {
            // Le system prompt vit dans `instructions`, pas dans input[].
            Role::System => {}
            Role::User => {
                let content = user_content(&msg.content);
                if !content.is_empty() {
                    input.push(json!({
                        "type": "message",
                        "role": "user",
                        "content": content,
                    }));
                }
            }
            Role::Assistant => assistant_items(&msg.content, &mut input),
            Role::Tool => tool_result_items(&msg.content, &mut input),
        }
    }
    Value::Array(input)
}

/// Blocs d'un message user → parts `input_text` / `input_image`.
fn user_content(blocks: &[ContentBlock]) -> Vec<Value> {
    let mut content = Vec::new();
    for b in blocks {
        match b {
            ContentBlock::Text { text } => {
                content.push(json!({ "type": "input_text", "text": text }));
            }
            ContentBlock::Image { media_type, data } => {
                content.push(json!({
                    "type": "input_image",
                    "detail": "auto",
                    "image_url": format!("data:{media_type};base64,{data}"),
                }));
            }
            // tool_use / tool_result ne sont pas portés par un message user.
            _ => {}
        }
    }
    content
}

/// Un message assistant produit : un item `message` (texte concaténé) puis un
/// item `function_call` par `tool_use`. Les blocs `thinking` ne sont PAS
/// réinjectés (MVP, cf. en-tête de module).
fn assistant_items(blocks: &[ContentBlock], input: &mut Vec<Value>) {
    let mut text = String::new();
    for b in blocks {
        if let ContentBlock::Text { text: t } = b {
            text.push_str(t);
        }
    }
    if !text.is_empty() {
        input.push(json!({
            "type": "message",
            "role": "assistant",
            "content": [ { "type": "output_text", "text": text, "annotations": [] } ],
        }));
    }
    for b in blocks {
        if let ContentBlock::ToolUse {
            id,
            name,
            input: args,
        } = b
        {
            input.push(json!({
                "type": "function_call",
                "call_id": id,
                "name": name,
                // arguments est une STRING JSON dans la Responses API.
                "arguments": args.to_string(),
            }));
        }
    }
}

/// Blocs `tool_result` (role Tool) → items `function_call_output`.
fn tool_result_items(blocks: &[ContentBlock], input: &mut Vec<Value>) {
    for b in blocks {
        if let ContentBlock::ToolResult {
            tool_use_id,
            content,
            ..
        } = b
        {
            input.push(json!({
                "type": "function_call_output",
                "call_id": tool_use_id,
                "output": content,
            }));
        }
    }
}

/// `ToolSpec` canonique → tool `function` plat de la Responses API. `strict: null`
/// reproduit le comportement Codex de Pi (≠ `false`, subtilité vérifiée).
fn build_tools(tools: &[ToolSpec]) -> Value {
    let arr: Vec<Value> = tools
        .iter()
        .map(|t| {
            json!({
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
                "strict": Value::Null,
            })
        })
        .collect();
    Value::Array(arr)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(messages: Vec<Message>, tools: Vec<ToolSpec>, system: Option<&str>) -> CanonicalRequest {
        CanonicalRequest {
            model: "gpt-5.4".into(),
            system: system.map(String::from),
            messages,
            tools,
            max_output_tokens: 4096,
        }
    }

    #[test]
    fn fixed_fields_are_present_and_store_is_false() {
        let body = build_responses_body(&req(vec![Message::user("salut")], vec![], None), None);
        assert_eq!(body["store"], json!(false));
        assert_eq!(body["stream"], json!(true));
        assert_eq!(body["model"], "gpt-5.4");
        assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
        assert_eq!(body["tool_choice"], "auto");
        assert_eq!(body["parallel_tool_calls"], json!(true));
        // pas de previous_response_id (SSE stateless).
        assert!(body.get("previous_response_id").is_none());
    }

    #[test]
    fn system_goes_to_instructions_not_input() {
        let body = build_responses_body(
            &req(vec![Message::user("hi")], vec![], Some("Tu es Numen.")),
            None,
        );
        assert_eq!(body["instructions"], "Tu es Numen.");
        // aucun item role:system dans input
        let input = body["input"].as_array().unwrap();
        assert!(input.iter().all(|i| i["role"] != "system"));
    }

    #[test]
    fn default_instructions_when_no_system() {
        let body = build_responses_body(&req(vec![Message::user("hi")], vec![], None), None);
        assert_eq!(body["instructions"], DEFAULT_INSTRUCTIONS);
    }

    #[test]
    fn user_text_maps_to_input_text_message() {
        let body = build_responses_body(&req(vec![Message::user("bonjour")], vec![], None), None);
        let item = &body["input"][0];
        assert_eq!(item["type"], "message");
        assert_eq!(item["role"], "user");
        assert_eq!(item["content"][0]["type"], "input_text");
        assert_eq!(item["content"][0]["text"], "bonjour");
    }

    #[test]
    fn assistant_tooluse_and_tool_result_correlate_by_call_id() {
        let assistant = Message::assistant(vec![
            ContentBlock::Text {
                text: "j'appelle".into(),
            },
            ContentBlock::ToolUse {
                id: "call_42".into(),
                name: "bash".into(),
                input: json!({ "cmd": "ls" }),
            },
        ]);
        let tool = Message::tool_result("call_42", "fichiers...", false);
        let body = build_responses_body(&req(vec![assistant, tool], vec![], None), None);
        let input = body["input"].as_array().unwrap();

        // message assistant (output_text) + function_call + function_call_output
        let msg = input.iter().find(|i| i["type"] == "message").unwrap();
        assert_eq!(msg["content"][0]["type"], "output_text");

        let fc = input.iter().find(|i| i["type"] == "function_call").unwrap();
        assert_eq!(fc["call_id"], "call_42");
        assert_eq!(fc["name"], "bash");
        // arguments est une STRING JSON.
        assert_eq!(fc["arguments"], "{\"cmd\":\"ls\"}");

        let out = input
            .iter()
            .find(|i| i["type"] == "function_call_output")
            .unwrap();
        assert_eq!(out["call_id"], "call_42");
        assert_eq!(out["output"], "fichiers...");
    }

    #[test]
    fn tools_map_to_flat_function_with_null_strict() {
        let spec = ToolSpec {
            name: "read".into(),
            description: "lit un fichier".into(),
            input_schema: json!({ "type": "object", "properties": { "path": {"type":"string"} } }),
        };
        let body = build_responses_body(&req(vec![Message::user("x")], vec![spec], None), None);
        let tool = &body["tools"][0];
        assert_eq!(tool["type"], "function");
        assert_eq!(tool["name"], "read");
        assert_eq!(tool["parameters"]["properties"]["path"]["type"], "string");
        // strict: null (≠ false) — fidélité Codex.
        assert!(tool["strict"].is_null());
    }

    #[test]
    fn no_tools_omits_tools_field() {
        let body = build_responses_body(&req(vec![Message::user("x")], vec![], None), None);
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn reasoning_effort_included_when_set_omitted_otherwise() {
        let with = build_responses_body(&req(vec![Message::user("x")], vec![], None), Some("high"));
        assert_eq!(with["reasoning"]["effort"], "high");
        assert_eq!(with["reasoning"]["summary"], "auto");
        let without = build_responses_body(&req(vec![Message::user("x")], vec![], None), None);
        assert!(without.get("reasoning").is_none());
    }

    #[test]
    fn assistant_text_and_calls_order() {
        // texte d'abord (message), puis function_call — comme Pi.
        let assistant = Message::assistant(vec![
            ContentBlock::ToolUse {
                id: "c1".into(),
                name: "a".into(),
                input: json!({}),
            },
            ContentBlock::Text {
                text: "après".into(),
            },
        ]);
        let body = build_responses_body(&req(vec![assistant], vec![], None), None);
        let input = body["input"].as_array().unwrap();
        assert_eq!(input[0]["type"], "message");
        assert_eq!(input[1]["type"], "function_call");
    }
}
