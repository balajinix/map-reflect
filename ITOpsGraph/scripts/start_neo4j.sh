#!/bin/bash
#
# start_neo4j.sh - Start Neo4j on local or remote machine
#
# Usage:
#   ./scripts/start_neo4j.sh local          # Start Neo4j locally via Docker
#   ./scripts/start_neo4j.sh remote         # SSH into remote and start Neo4j (systemctl)
#   ./scripts/start_neo4j.sh status         # Check status of both instances
#   ./scripts/start_neo4j.sh stop [local|remote]  # Stop Neo4j
#
# Neo4j instances:
#   Local:  bolt://localhost:7687        Browser: http://localhost:7474
#   Remote: bolt://ai4graphs:7687        Browser: http://ai4graphs.vpc.cloud9.ibm.com:7474

set -e

# ── Configuration ────────────────────────────────────────────────────────────
NEO4J_IMAGE="neo4j:5"
CONTAINER_NAME="neo4j-itopsgraph"
NEO4J_PASSWORD="@i4graphs"
NEO4J_HTTP_PORT=7474
NEO4J_BOLT_PORT=7687

REMOTE_HOST="9.47.155.20"
REMOTE_USER="bganesa1"
REMOTE_NEO4J_SERVICE="neo4j"  # systemd service name

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ── Helper functions ─────────────────────────────────────────────────────────
info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }

wait_for_neo4j() {
    local host="$1"
    local port="${2:-$NEO4J_HTTP_PORT}"
    local max_wait=60
    local elapsed=0

    info "Waiting for Neo4j to be ready at ${host}:${port}..."
    while [ $elapsed -lt $max_wait ]; do
        if curl -s "http://${host}:${port}" >/dev/null 2>&1; then
            ok "Neo4j is ready!"
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
        printf "."
    done
    echo ""
    err "Neo4j did not become ready within ${max_wait}s"
    return 1
}

# ── Docker run command (for local) ───────────────────────────────────────────
neo4j_docker_run_cmd() {
    local data_dir="$1"
    cat <<DOCKER_CMD
docker run -d \\
  --name ${CONTAINER_NAME} \\
  --restart unless-stopped \\
  -p ${NEO4J_HTTP_PORT}:7474 \\
  -p ${NEO4J_BOLT_PORT}:7687 \\
  -v ${data_dir}/data:/data \\
  -v ${data_dir}/logs:/logs \\
  -v ${data_dir}/plugins:/plugins \\
  -e NEO4J_AUTH=neo4j/${NEO4J_PASSWORD} \\
  -e NEO4J_PLUGINS='["apoc"]' \\
  -e NEO4J_server_memory_heap_initial__size=512m \\
  -e NEO4J_server_memory_heap_max__size=1G \\
  -e NEO4J_server_memory_pagecache_size=512m \\
  -e NEO4J_dbms_security_procedures_unrestricted='apoc.*' \\
  ${NEO4J_IMAGE}
DOCKER_CMD
}

# ── Local ────────────────────────────────────────────────────────────────────
start_local() {
    info "Starting Neo4j locally..."

    # Check Docker is available
    if ! command -v docker &>/dev/null; then
        err "Docker is not installed or not in PATH"
        exit 1
    fi

    # Check if container already exists
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            ok "Neo4j container '${CONTAINER_NAME}' is already running"
            info "Browser: http://localhost:${NEO4J_HTTP_PORT}"
            info "Bolt:    bolt://localhost:${NEO4J_BOLT_PORT}"
            return 0
        else
            info "Container exists but is stopped. Starting it..."
            docker start "${CONTAINER_NAME}"
            wait_for_neo4j "localhost"
            info "Browser: http://localhost:${NEO4J_HTTP_PORT}"
            info "Bolt:    bolt://localhost:${NEO4J_BOLT_PORT}"
            return 0
        fi
    fi

    # Create local data directory
    local data_dir="${HOME}/neo4j-data"
    mkdir -p "${data_dir}/data" "${data_dir}/logs" "${data_dir}/plugins"

    info "Data directory: ${data_dir}"
    info "Pulling ${NEO4J_IMAGE} if needed..."

    # Run the container
    eval "$(neo4j_docker_run_cmd "${data_dir}")"

    wait_for_neo4j "localhost"

    echo ""
    ok "Neo4j is running locally"
    info "Browser: http://localhost:${NEO4J_HTTP_PORT}"
    info "Bolt:    bolt://localhost:${NEO4J_BOLT_PORT}"
    info "Auth:    neo4j / ${NEO4J_PASSWORD}"
}

# ── Remote (native install via systemctl) ────────────────────────────────────
start_remote() {
    info "Starting Neo4j on remote host ${REMOTE_USER}@${REMOTE_HOST} via systemctl..."

    # Check SSH connectivity
    if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "${REMOTE_USER}@${REMOTE_HOST}" "echo ok" >/dev/null 2>&1; then
        err "Cannot SSH into ${REMOTE_USER}@${REMOTE_HOST}"
        err "Ensure SSH keys are set up and you have network access (VPN?)"
        exit 1
    fi

    ok "SSH connection verified"

    ssh -t "${REMOTE_USER}@${REMOTE_HOST}" bash <<REMOTE_SCRIPT
set -e

# Check current status
STATUS=\$(systemctl is-active ${REMOTE_NEO4J_SERVICE} 2>/dev/null || true)

if [ "\$STATUS" = "active" ]; then
    echo "Neo4j service is already running"
    systemctl status ${REMOTE_NEO4J_SERVICE} --no-pager -l 2>/dev/null | head -5
    exit 0
fi

echo "Starting Neo4j service..."
sudo systemctl start ${REMOTE_NEO4J_SERVICE}

echo "Enabling Neo4j to start on boot..."
sudo systemctl enable ${REMOTE_NEO4J_SERVICE} 2>/dev/null || true

echo "Neo4j service started."
systemctl status ${REMOTE_NEO4J_SERVICE} --no-pager -l 2>/dev/null | head -5
REMOTE_SCRIPT

    wait_for_neo4j "ai4graphs.vpc.cloud9.ibm.com"

    echo ""
    ok "Neo4j is running on remote host"
    info "Browser: http://ai4graphs.vpc.cloud9.ibm.com:${NEO4J_HTTP_PORT}/browser/"
    info "Bolt:    bolt://ai4graphs.vpc.cloud9.ibm.com:${NEO4J_BOLT_PORT}"
    info "Auth:    neo4j / ${NEO4J_PASSWORD}"
}

# ── Status ───────────────────────────────────────────────────────────────────
check_status() {
    echo -e "${BLUE}═══════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Neo4j Instance Status${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════${NC}"
    echo ""

    # Local
    echo -e "${YELLOW}Local (localhost):${NC}"
    if command -v docker &>/dev/null; then
        if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${CONTAINER_NAME}$"; then
            ok "Container '${CONTAINER_NAME}' is RUNNING"
            docker ps --filter "name=${CONTAINER_NAME}" --format "  Image: {{.Image}}  Uptime: {{.Status}}  Ports: {{.Ports}}"
        elif docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q "^${CONTAINER_NAME}$"; then
            warn "Container '${CONTAINER_NAME}' exists but is STOPPED"
        else
            warn "No Neo4j container found"
        fi
    else
        warn "Docker not available locally"
    fi

    # Check bolt connectivity
    if curl -s "http://localhost:${NEO4J_HTTP_PORT}" >/dev/null 2>&1; then
        ok "HTTP endpoint responding on localhost:${NEO4J_HTTP_PORT}"
    else
        warn "HTTP endpoint not reachable on localhost:${NEO4J_HTTP_PORT}"
    fi
    echo ""

    # Remote
    echo -e "${YELLOW}Remote (ai4graphs.vpc.cloud9.ibm.com):${NC}"
    if ssh -o ConnectTimeout=5 -o BatchMode=yes "${REMOTE_USER}@${REMOTE_HOST}" "echo ok" >/dev/null 2>&1; then
        ok "SSH connection OK"
        ssh "${REMOTE_USER}@${REMOTE_HOST}" bash <<REMOTE_STATUS
STATUS=\$(systemctl is-active ${REMOTE_NEO4J_SERVICE} 2>/dev/null || true)
if [ "\$STATUS" = "active" ]; then
    echo "  [OK]   Neo4j service is RUNNING"
    systemctl status ${REMOTE_NEO4J_SERVICE} --no-pager -l 2>/dev/null | grep -E "(Active:|Main PID:)" | sed 's/^/  /'
else
    echo "  [WARN] Neo4j service is \$STATUS"
fi
REMOTE_STATUS
    else
        warn "Cannot SSH into remote host (check VPN/network)"
    fi

    if curl -s --connect-timeout 5 "http://ai4graphs.vpc.cloud9.ibm.com:${NEO4J_HTTP_PORT}" >/dev/null 2>&1; then
        ok "HTTP endpoint responding on ai4graphs:${NEO4J_HTTP_PORT}"
    else
        warn "HTTP endpoint not reachable on ai4graphs:${NEO4J_HTTP_PORT}"
    fi
}

# ── Stop ─────────────────────────────────────────────────────────────────────
stop_neo4j() {
    local target="${1:-both}"

    if [ "$target" = "local" ] || [ "$target" = "both" ]; then
        info "Stopping local Neo4j..."
        if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${CONTAINER_NAME}$"; then
            docker stop "${CONTAINER_NAME}"
            ok "Local Neo4j stopped"
        else
            warn "No running local Neo4j container found"
        fi
    fi

    if [ "$target" = "remote" ] || [ "$target" = "both" ]; then
        info "Stopping remote Neo4j..."
        if ssh -o ConnectTimeout=5 -o BatchMode=yes "${REMOTE_USER}@${REMOTE_HOST}" "echo ok" >/dev/null 2>&1; then
            ssh -t "${REMOTE_USER}@${REMOTE_HOST}" bash <<REMOTE_STOP
STATUS=\$(systemctl is-active ${REMOTE_NEO4J_SERVICE} 2>/dev/null || true)
if [ "\$STATUS" = "active" ]; then
    sudo systemctl stop ${REMOTE_NEO4J_SERVICE}
    echo "Remote Neo4j stopped"
else
    echo "Neo4j service is already \$STATUS"
fi
REMOTE_STOP
            ok "Remote Neo4j stopped"
        else
            err "Cannot SSH into remote host"
        fi
    fi
}

# ── Main ─────────────────────────────────────────────────────────────────────
usage() {
    echo "Usage: $0 <command> [target]"
    echo ""
    echo "Commands:"
    echo "  local          Start Neo4j locally via Docker"
    echo "  remote         Start Neo4j on remote host via SSH + systemctl"
    echo "  both           Start on both local and remote"
    echo "  status         Check status of both instances"
    echo "  stop [target]  Stop Neo4j (target: local, remote, or both)"
    echo ""
    echo "Examples:"
    echo "  $0 local"
    echo "  $0 remote"
    echo "  $0 both"
    echo "  $0 status"
    echo "  $0 stop local"
    exit 1
}

case "${1:-}" in
    local)
        start_local
        ;;
    remote)
        start_remote
        ;;
    both)
        start_local
        echo ""
        start_remote
        ;;
    status)
        check_status
        ;;
    stop)
        stop_neo4j "${2:-both}"
        ;;
    -h|--help|"")
        usage
        ;;
    *)
        err "Unknown command: $1"
        usage
        ;;
esac
