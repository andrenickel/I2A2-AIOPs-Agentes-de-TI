import { memo } from 'react';

const Footer = memo(() => {
  return (
    <footer className="bg-white/10 backdrop-blur-md border-t border-white/20 mt-16">
      <div className="container mx-auto px-4 py-8">
        <div className="text-center text-white/80">
          <p>&copy; 2025 <strong>AIOps Fiscal Agent</strong> — Automação e Inteligência Fiscal sob o comando da IA.</p>
        </div>
      </div>
    </footer>
  );
});

Footer.displayName = 'Footer';

export default Footer;
