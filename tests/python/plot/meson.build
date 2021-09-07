e = find_program('./test-run.py', required: true)
test('plot', e,
      timeout : 100,
      suite: 'python',
      workdir: meson.current_source_dir(),
      depends: python_extension
    )
