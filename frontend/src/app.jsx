import React from 'react'
import Chat from './chat'

export default function App(){
  return (
    <div style={{padding:24, display:'flex', justifyContent:'center'}}>
      <div style={{width:720, border:'1px solid #eee', borderRadius:12, padding:18}}>
        <h2>Medical AI Chatbot</h2>
        <Chat />
      </div>
    </div>
  )
}

