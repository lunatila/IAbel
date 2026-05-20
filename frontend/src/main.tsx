import React, { useState, useEffect } from 'react'
import ReactDOM from 'react-dom/client'
import { Toaster } from 'react-hot-toast'
import Layout from './components/layout/Layout'
import ChatInterface from './components/chat/ChatInterface'
import DocumentSearch from './components/search/DocumentSearch'
import PDFUpload from './components/upload/PDFUpload'
import SystemStatus from './components/dashboard/SystemStatus'
import { useAppStore } from './stores/appStore'
import { apiService } from './services/api'
import './index.css'

const App: React.FC = () => {
  const [currentPage, setCurrentPage] = useState('chat')
  const { setSystemStatus } = useAppStore()

  // Load system status on mount
  useEffect(() => {
    const loadSystemStatus = async () => {
      try {
        const status = await apiService.getSystemStatus()
        setSystemStatus(status)
      } catch (error) {
        console.error('Failed to load system status:', error)
      }
    }

    loadSystemStatus()
    // Refresh status every 30 seconds
    const interval = setInterval(loadSystemStatus, 30000)
    return () => clearInterval(interval)
  }, [setSystemStatus])

  const renderPage = () => {
    switch (currentPage) {
      case 'chat':
        return <ChatInterface />
      case 'search':
        return <DocumentSearch />
      case 'upload':
        return <PDFUpload />
      case 'status':
        return <SystemStatus />
      default:
        return <ChatInterface />
    }
  }

  return (
    <>
      <Layout currentPage={currentPage} onPageChange={setCurrentPage}>
        {renderPage()}
      </Layout>
      <Toaster 
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: 'var(--toast-bg)',
            color: 'var(--toast-color)',
            border: '1px solid var(--toast-border)',
          },
        }}
      />
    </>
  )
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)