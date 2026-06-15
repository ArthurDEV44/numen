//! `agent-sandbox` — bac à sable d'exécution (US-020). Deux protections
//! complémentaires :
//! - **FS** : confinement kernel-level via Landlock (`fs`), appliqué process-wide
//!   au démarrage → toute écriture est confinée au workspace (agent ET
//!   sous-process Bash hérités).
//! - **Réseau** : proxy CONNECT allow-list (`proxy`) ; les sous-process outils
//!   reçoivent `HTTP(S)_PROXY` → filtrage best-effort par hostname.
//!
//! Linux-first : hors Linux, le FS dégrade explicitement (AC3). Le proxy reste
//! disponible (pur tokio).
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod fs;
pub mod proxy;

pub use fs::{SandboxError, SandboxStatus, enforce_process};
pub use proxy::{ProxyHandle, ProxyPolicy, spawn as spawn_proxy};

/// Injecte la variable d'environnement proxy sur la commande d'un outil (Bash),
/// SANS toucher l'environnement global du process (le provider de l'agent
/// continue d'appeler le réseau en direct). À utiliser dans le closure de
/// durcissement passé au `ToolCtx`.
pub fn set_proxy_env(cmd: &mut tokio::process::Command, proxy_addr: &str) {
    let url = format!("http://{proxy_addr}");
    cmd.env("HTTP_PROXY", &url)
        .env("HTTPS_PROXY", &url)
        .env("http_proxy", &url)
        .env("https_proxy", &url)
        // empêche les outils de bypasser le proxy pour localhost only si voulu —
        // on laisse NO_PROXY vide (tout passe par le proxy filtrant).
        .env("NO_PROXY", "");
}
