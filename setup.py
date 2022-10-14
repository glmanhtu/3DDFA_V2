from setuptools import setup, Extension

if __name__ == '__main__':
    setup_args = dict(
        ext_modules=[
            Extension(
                'Sim3DR_Cython',
                ['lib/mymodule.c', 'lib/mypackage.c', 'lib/myalloc.c'],
                include_dirs=['lib'],
                py_limited_api=True
            )
        ]
    )
    setup(**setup_args)

    setup()
