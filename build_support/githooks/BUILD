shell_sources(name="scripts", sources=["install_hooks.sh", "pre-commit"])

run_shell_command(
    name="install_hooks",
    command="./install_hooks.sh",
    execution_dependencies=[":scripts"],
)
