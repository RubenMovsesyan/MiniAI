// Shared Plotly dark theme + sidebar nav for the PyTorch-track docs.
// Same mechanism as ~/Documents/ml_training. Grows as pages are added: append
// to PAGES.
function darkLayout(extra){
  const base = {
    paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
    font:{color:'#c9d1d9', family:'-apple-system,Segoe UI,Roboto,sans-serif', size:13},
    xaxis:{gridcolor:'#30363d', zerolinecolor:'#30363d', linecolor:'#8b949e'},
    yaxis:{gridcolor:'#30363d', zerolinecolor:'#30363d', linecolor:'#8b949e'},
    legend:{bgcolor:'rgba(22,27,34,0.6)'},
    margin:{t:30,r:20,b:55,l:70}
  };
  return Object.assign(base, extra||{});
}
const CFG = {responsive:true, displayModeBar:false};
const C = {
  blue:'#58a6ff', green:'#3fb950', orange:'#d29922', red:'#f85149',
  purple:'#bc8cff', pink:'#f778ba', dim:'#8b949e'
};

// ── The target curve you must reproduce in PyTorch.
// Measured on the C++/CUDA repo: 784 -> 128 (ReLU, He) -> 10, softmax-CE,
// SGD lr 0.1, batch 100, seed 42, MNIST 60k/10k. loss = mean train CE,
// acc = test %. (Same numbers the math docs plot.)
const EPOCHS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15];
const TARGET = {
  loss:[0.5309,0.2239,0.1726,0.1413,0.1256,0.1303,0.1109,0.0957,0.0922,0.0791,0.0835,0.0486,0.0671,0.054,0.0538],
  acc: [92.5,94.03,95.04,95.6,96.18,96.42,96.75,96.97,97.06,97.23,97.32,97.41,97.4,97.53,97.6]
};

const PAGES = [
  ['index.html',           'Overview'],
  ['pytorch_guide.html',   'PyTorch: match the C++ model'],
  ['onnx_and_engine.html', 'ONNX &amp; the inference engine']
];
function buildNav(current, sections){
  const el = document.querySelector('nav.toc');
  let h = '<h2>PyTorch track</h2>';
  PAGES.forEach(([href,label],i) => {
    const cls = href === current ? 'doclink here' : 'doclink';
    const mark = href === current ? '▸ ' : '';
    h += `<a class="${cls}" href="${href}">${mark}${i+1}. ${label}</a>`;
  });
  h += '<h2>This page</h2>';
  (sections||[]).forEach(([id,label]) => { h += `<a href="#${id}">${label}</a>`; });
  el.innerHTML = h;
}
