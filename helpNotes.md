## Install Errors

### Cannot install LightGBM

For mac:
LightGBM expects by default to find OpenMP installed locally, you can install it like this:

<pre id="code-block-1">
<code>
brew install libomp
</code>
</pre>
<button onclick="copyToClipboard('#code-block-1')">Copy Command</button>

Alternatively, if you don't want to use OpenMP, you can install a version of LightGBM that does not use OpenMP. Training and prediction will be slower (as a result of not using multithreading), but you'll be able to build LightGBM.

<pre id="code-block-2">
<code>
pip install --upgrade pip
pip install \
   --no-binary lightgbm \
   --config-settings=cmake.define.USE_OPENMP=OFF \
   'lightgbm>=4.0.0'
</code>
</pre>
<button onclick="copyToClipboard('#code-block-2')">Copy Command</button>

### Page Link: [source](https://github.com/microsoft/LightGBM/issues/6035)