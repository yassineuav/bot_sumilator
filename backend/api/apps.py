from django.apps import AppConfig


class ApiConfig(AppConfig):
    name = 'api'

    def ready(self):
        import os
        # Prevent multiple starts in auto-reload mode
        if os.environ.get('RUN_MAIN') == 'true':
            import sys
            # Add core to path
            sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core'))
            from core.scheduler import start_scheduler
            start_scheduler()
