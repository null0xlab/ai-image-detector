import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'

function scrollToHash() {
  const hash = window.location.hash
  if (!hash) return
  requestAnimationFrame(() => {
    document.querySelector(hash)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  })
}

window.addEventListener('hashchange', scrollToHash)
scrollToHash()

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
