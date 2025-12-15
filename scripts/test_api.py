import importlib.util
import sys
from pathlib import Path
import json

def import_local_main():
    path = Path('app') / 'main.py'
    spec = importlib.util.spec_from_file_location('local_app_main', str(path))
    module = importlib.util.module_from_spec(spec)
    # Ensure local 'app' package modules are importable by loading them first
    def load_local(name, relpath):
        p = Path('app') / relpath
        s = importlib.util.spec_from_file_location(name, str(p))
        mod = importlib.util.module_from_spec(s)
        sys.modules[name] = mod
        s.loader.exec_module(mod)
        return mod
    load_local('app.schemas', 'schemas.py')
    load_local('app.utils', 'utils.py')
    sys.modules['local_app_main'] = module
    spec.loader.exec_module(module)
    return module


def main():
    m = import_local_main()
    # predict-demand
    req = type('Req', (), {'hour': 18, 'temperature': 32.0, 'voltage': 230.0})
    print('calling predict_demand...')
    print(m.predict_demand(req))
    print('calling peak_hour...')
    print(m.peak_hour(req))

if __name__ == '__main__':
    main()
