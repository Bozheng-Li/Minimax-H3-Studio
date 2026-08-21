#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$project_dir/data/state" "$project_dir/data/assets" "$project_dir/data/logs" "$project_dir/data/checkpoints" "$project_dir/outputs"

move_if_present() {
  local source="$1" destination="$2"
  if [[ -e "$source" && ! -e "$destination" ]]; then
    mkdir -p "$(dirname "$destination")"
    mv "$source" "$destination"
    printf 'moved %s -> %s\n' "${source#$project_dir/}" "${destination#$project_dir/}"
  fi
}

move_if_present "$project_dir/.h3_frontend_jobs.json" "$project_dir/data/state/jobs.json"
move_if_present "$project_dir/.h3_model_state.json" "$project_dir/data/state/model_state.json"
move_if_present "$project_dir/.h3_director_projects.json" "$project_dir/data/state/director_projects.json"
move_if_present "$project_dir/.h3_media_library.json" "$project_dir/data/state/media_library.json"
move_if_present "$project_dir/.h3_queue_assets" "$project_dir/data/assets/queue"
move_if_present "$project_dir/.h3_director_assets" "$project_dir/data/assets/director"
move_if_present "$project_dir/.h3_media_library" "$project_dir/data/assets/library"
move_if_present "$project_dir/.h3_latent_checkpoints" "$project_dir/data/checkpoints/latents"
move_if_present "$project_dir/.h3_backend.log" "$project_dir/data/logs/h3_backend.log"

shopt -s nullglob
for file in "$project_dir"/*.mp4; do
  move_if_present "$file" "$project_dir/outputs/$(basename "$file")"
done
printf 'H3 Studio layout migration complete.\n'
