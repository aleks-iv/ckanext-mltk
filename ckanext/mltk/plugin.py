from typing import Any

from ckan import plugins as p
from ckan.plugins import toolkit as tk


@tk.blanket.actions
@tk.blanket.config_declarations
class MltkPlugin(p.SingletonPlugin):
    p.implements(p.IConfigurer)

    # IConfigurer

    def update_config(self, config_: Any):
        tk.add_template_directory(config_, "templates")
        tk.add_public_directory(config_, "public")
        tk.add_resource("assets", "mltk")
