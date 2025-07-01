import subprocess

def run_script(script_name):
    print(f"Running {script_name}...")
    try:
        subprocess.run(["python3", script_name], check=True)
    except subprocess.CalledProcessError:
        print(f"Error running {script_name}")

def main():
    scripts = [
        "get_pitcher_data.py",
        "merged_props.py",
        "find_best_lines.py",
        "post_game_evaluation.py"
    ]

    for script in scripts:
        run_script(script)

    print("\ncompleted.")

if __name__ == "__main__":
    main()
