import { memo } from 'react';
import { NavLink } from 'react-router-dom';
import { X } from 'lucide-react';
import type { NavigationItem } from '@/hooks/useNavigation';

interface MobileSidebarProps {
  navigation: NavigationItem[];
  isOpen: boolean;
  onClose: () => void;
}

const MobileSidebar = memo(({ navigation, isOpen, onClose }: MobileSidebarProps) => {
  if (!isOpen) return null;

  return (
    <div className="md:hidden fixed inset-0 z-50 bg-black/50" onClick={onClose}>
      <div 
        className="fixed left-0 top-0 h-full w-64 bg-white/95 backdrop-blur-md p-6" 
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-8">
          <h2 className="text-lg font-semibold">Menu</h2>
          <button onClick={onClose} aria-label="Close menu">
            <X className="w-6 h-6" />
          </button>
        </div>
        <nav className="space-y-4">
          {navigation.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              end={item.path === '/'}
              onClick={onClose}
              className={({ isActive }) =>
                `flex items-center space-x-3 w-full px-3 py-2 rounded-lg transition-colors ${
                  isActive
                    ? 'bg-primary/10 text-primary'
                    : 'text-foreground hover:bg-muted'
                }`
              }
            >
              <item.icon className="w-5 h-5" />
              <span>{item.name}</span>
            </NavLink>
          ))}
        </nav>
      </div>
    </div>
  );
});

MobileSidebar.displayName = 'MobileSidebar';

export default MobileSidebar;
