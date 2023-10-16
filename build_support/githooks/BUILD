shell_sources(name="scripts", sources=["install_hooks.sh", "pre-commit.sh"])

run_shell_command(
    name="install_hooks",
    command="./install_hooks.sh",
    # tools=["git", "bash", "set"],
    execution_dependencies=[":scripts"],
)
