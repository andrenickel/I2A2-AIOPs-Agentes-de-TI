import { memo } from 'react';
import { NavLink } from 'react-router-dom';
import { FileText, Menu, X } from 'lucide-react';
import { NotificationSystem } from '../NotificationSystem';
import type { NavigationItem } from '@/hooks/useNavigation';

interface HeaderProps {
  navigation: NavigationItem[];
  sidebarOpen: boolean;
  onToggleSidebar: () => void;
}

const Header = memo(({ navigation, sidebarOpen, onToggleSidebar }: HeaderProps) => {
  return (
    <header className="bg-white/10 backdrop-blur-md border-b border-white/20">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-12 h-12 bg-white/20 rounded-lg flex items-center justify-center">
              <FileText className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">AIOps Fiscal Agent</h1>
              <p className="text-sm text-white/80">Automatização de Documentos Fiscais</p>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <NotificationSystem />

            <nav className="hidden md:flex space-x-6">
              {navigation.map((item) => (
                <NavLink
                  key={item.path}
                  to={item.path}
                  end={item.path === '/'}
                  data-tour={item.name.toLowerCase()}
                  className={({ isActive }) =>
                    `flex items-center space-x-2 px-3 py-2 rounded-lg transition-colors ${
                      isActive
                        ? 'bg-white/20 text-white'
                        : 'text-white/80 hover:text-white hover:bg-white/10'
                    }`
                  }
                >
                  <item.icon className="w-4 h-4" />
                  <span>{item.name}</span>
                </NavLink>
              ))}
            </nav>

            <button
              onClick={onToggleSidebar}
              className="md:hidden text-white p-2"
              aria-label="Toggle menu"
            >
              {sidebarOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
          </div>
        </div>
      </div>
    </header>
  );
});

Header.displayName = 'Header';

export default Header;
