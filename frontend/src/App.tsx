import { lazy, Suspense } from 'react';
import { Routes, Route } from 'react-router-dom';
import { useNavigation } from './hooks/useNavigation';
import Header from './components/layout/Header';
import MobileSidebar from './components/layout/MobileSidebar';
import Footer from './components/layout/Footer';
import HomePage from './components/home/HomePage';
import NotFound from './pages/NotFound';
import { ProductTour, TourButton } from './components/onboarding/ProductTour';

const Dashboard = lazy(() => import('./components/Dashboard'));
const FileUpload = lazy(() => import('./components/FileUpload'));
const Chat = lazy(() => import('./components/Chat'));
const DocumentsPage = lazy(() => import('./components/documents/DocumentsPage'));

function App() {
  const { sidebarOpen, navigation, toggleSidebar, setSidebarOpen } = useNavigation();

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary to-secondary">
      <ProductTour />
      <TourButton />
      
      <Header
        navigation={navigation}
        sidebarOpen={sidebarOpen}
        onToggleSidebar={toggleSidebar}
      />

      <MobileSidebar
        navigation={navigation}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      <main className="container mx-auto px-4 py-8">
        <Suspense fallback={<div className="text-white text-center">Carregando...</div>}>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/chat" element={<Chat />} />
            <Route path="/upload" element={<FileUpload />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/documents" element={<DocumentsPage />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </Suspense>
      </main>

      <Footer />
    </div>
  );
}

export default App;
