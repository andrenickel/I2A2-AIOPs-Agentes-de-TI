import { useState, useCallback } from 'react';
import { Home, Upload, BarChart3, FileText, Bot } from 'lucide-react';

export interface NavigationItem {
  name: string;
  path: string;
  icon: typeof Home;
}

export const useNavigation = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const navigation: NavigationItem[] = [
    { name: 'Início', path: '/', icon: Home },
    { name: 'Chat', path: '/chat', icon: Bot },
    { name: 'Upload', path: '/upload', icon: Upload },
    { name: 'Dashboard', path: '/dashboard', icon: BarChart3 },
    { name: 'Documentos', path: '/documents', icon: FileText },
  ];

  const toggleSidebar = useCallback(() => {
    setSidebarOpen(prev => !prev);
  }, []);

  return {
    sidebarOpen,
    navigation,
    toggleSidebar,
    setSidebarOpen,
  };
};
