
from dynaconf import Dynaconf

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=[                      # Paths or globs to any toml|yaml|ini|json|py
        "configs/default_settings.toml",  # a file for default settings
        "configs/settings.toml",          # a file for main settings
        "configs/.secrets.toml"           # a file for sensitive data (gitignored)
    ])

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.
