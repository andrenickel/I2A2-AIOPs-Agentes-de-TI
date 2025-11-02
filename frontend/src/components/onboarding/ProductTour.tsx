import { useState, useEffect } from 'react';
import Joyride, { Step, CallBackProps, STATUS } from 'react-joyride';
import { useLocation } from 'react-router-dom';

const steps: Step[] = [
  {
    target: 'body',
    content: 'Bem-vindo ao AIOps Fiscal Agent! Vamos fazer um tour rápido pelas principais funcionalidades.',
    placement: 'center',
    disableBeacon: true,
  },
  {
    target: '[data-tour="chat"]',
    content: 'Use o Chat com IA para tirar dúvidas sobre seus documentos fiscais e obter insights.',
    placement: 'bottom',
  },
  {
    target: '[data-tour="upload"]',
    content: 'Aqui você pode fazer upload de seus arquivos XML de NF-e para processamento automático.',
    placement: 'bottom',
  },
  {
    target: '[data-tour="dashboard"]',
    content: 'O Dashboard apresenta estatísticas e relatórios detalhados sobre seus documentos.',
    placement: 'bottom',
  },
  {
    target: '[data-tour="documentos"]',
    content: 'Acesse todos os seus documentos processados e gerencie-os facilmente.',
    placement: 'bottom',
  },
];

const homeSteps: Step[] = [
  {
    target: '[data-tour="hero-cta"]',
    content: 'Clique aqui para começar a processar seus primeiros documentos XML gratuitamente!',
    placement: 'bottom',
  },
  {
    target: '[data-tour="features"]',
    content: 'Conheça as principais funcionalidades: Upload, Validação Inteligente e Relatórios Automáticos.',
    placement: 'top',
  },
  {
    target: '[data-tour="stats"]',
    content: 'Acompanhe estatísticas em tempo real sobre documentos processados e taxa de sucesso.',
    placement: 'top',
  },
];

const dashboardSteps: Step[] = [
  {
    target: '[data-tour="filters"]',
    content: 'Use os filtros para visualizar dados de diferentes períodos.',
    placement: 'bottom',
  },
  {
    target: '[data-tour="kpi-cards"]',
    content: 'Acompanhe os principais indicadores: documentos, valores, erros e taxa de processamento.',
    placement: 'bottom',
  },
  {
    target: '[data-tour="charts"]',
    content: 'Visualize gráficos interativos com a evolução dos seus dados. Clique nos cards para expandir!',
    placement: 'top',
  },
  {
    target: '[data-tour="goals"]',
    content: 'Defina e acompanhe suas metas mensais de processamento e qualidade.',
    placement: 'left',
  },
];

interface ProductTourProps {
  onComplete?: () => void;
}

export function ProductTour({ onComplete }: ProductTourProps) {
  const [run, setRun] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);
  const location = useLocation();
  const [currentSteps, setCurrentSteps] = useState<Step[]>(steps);

  useEffect(() => {
    // Check if user has seen the tour
    const hasSeenTour = localStorage.getItem('hasSeenProductTour');
    
    if (!hasSeenTour) {
      // Wait 5 seconds for the page to load and user to see the content
      const timer = setTimeout(() => {
        setRun(true);
      }, 5000);
      
      return () => clearTimeout(timer);
    }
  }, []);

  useEffect(() => {
    // Update steps based on current route
    if (location.pathname === '/') {
      setCurrentSteps([...steps, ...homeSteps]);
    } else if (location.pathname === '/dashboard') {
      setCurrentSteps([...steps, ...dashboardSteps]);
    } else {
      setCurrentSteps(steps);
    }
  }, [location.pathname]);

  const handleJoyrideCallback = (data: CallBackProps) => {
    const { status, index, type } = data;
    
    if (([STATUS.FINISHED, STATUS.SKIPPED] as string[]).includes(status)) {
      setRun(false);
      localStorage.setItem('hasSeenProductTour', 'true');
      onComplete?.();
    }

    if (type === 'step:after') {
      setStepIndex(index + 1);
    }
  };

  const restartTour = () => {
    setStepIndex(0);
    setRun(true);
  };

  // Expose restart function globally
  useEffect(() => {
    (window as any).restartProductTour = restartTour;
    return () => {
      delete (window as any).restartProductTour;
    };
  }, []);

  return (
    <Joyride
      steps={currentSteps}
      run={run}
      stepIndex={stepIndex}
      continuous
      showProgress
      showSkipButton
      spotlightPadding={8}
      disableOverlayClose
      disableCloseOnEsc
      disableScrolling
      callback={handleJoyrideCallback}
      styles={{
        options: {
          primaryColor: 'hsl(147, 50%, 36%)',
          textColor: 'hsl(0, 0%, 100%)',
          backgroundColor: 'rgba(20, 20, 30, 0.95)',
          arrowColor: 'rgba(20, 20, 30, 0.95)',
          overlayColor: 'rgba(0, 0, 0, 0.4)',
          zIndex: 10000,
        },
        spotlight: {
          borderRadius: '16px',
          boxShadow: '0 0 0 9999px rgba(0, 0, 0, 0.75), 0 0 30px rgba(147, 197, 114, 0.5)',
        },
        tooltip: {
          borderRadius: '20px',
          padding: '0',
          background: 'linear-gradient(135deg, rgba(20, 20, 30, 0.98) 0%, rgba(30, 30, 50, 0.98) 100%)',
          backdropFilter: 'blur(20px)',
          border: '1px solid rgba(147, 197, 114, 0.3)',
          boxShadow: '0 20px 60px rgba(0, 0, 0, 0.5), 0 0 40px rgba(147, 197, 114, 0.2)',
        },
        tooltipContainer: {
          textAlign: 'left',
          padding: '28px',
        },
        tooltipTitle: {
          fontSize: '22px',
          fontWeight: '700',
          marginBottom: '12px',
          color: 'hsl(147, 50%, 60%)',
          textShadow: '0 0 20px rgba(147, 197, 114, 0.5)',
        },
        tooltipContent: {
          fontSize: '16px',
          lineHeight: '1.8',
          color: 'rgba(255, 255, 255, 0.9)',
          marginBottom: '20px',
        },
        buttonNext: {
          background: 'linear-gradient(135deg, hsl(147, 50%, 36%) 0%, hsl(147, 50%, 46%) 100%)',
          color: 'white',
          borderRadius: '12px',
          padding: '12px 24px',
          fontSize: '14px',
          fontWeight: '600',
          border: 'none',
          boxShadow: '0 4px 15px rgba(147, 197, 114, 0.4)',
          transition: 'all 0.3s ease',
        },
        buttonBack: {
          color: 'rgba(255, 255, 255, 0.7)',
          marginRight: '12px',
          background: 'rgba(255, 255, 255, 0.1)',
          borderRadius: '12px',
          padding: '12px 20px',
          border: '1px solid rgba(255, 255, 255, 0.2)',
          transition: 'all 0.3s ease',
        },
        buttonSkip: {
          color: 'rgba(255, 255, 255, 0.7)',
          fontSize: '14px',
          padding: '8px 16px',
          background: 'transparent',
          border: 'none',
          textDecoration: 'none',
          marginRight: 'auto',
          fontWeight: '500',
          transition: 'all 0.3s ease',
        },
      }}
      locale={{
        back: 'Voltar',
        close: 'Fechar',
        last: 'Finalizar',
        next: 'Próximo',
        skip: 'Pular tour',
      }}
    />
  );
}

// Helper component to show tour button
export function TourButton() {
  const [isHovered, setIsHovered] = useState(false);

  const handleClick = () => {
    localStorage.removeItem('hasSeenProductTour');
    if ((window as any).restartProductTour) {
      (window as any).restartProductTour();
    }
  };

  return (
    <button
      onClick={handleClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      className="fixed bottom-6 right-6 z-50 group"
      title="Iniciar tour guiado"
    >
      <div className="relative">
        {/* Glow effect */}
        <div className="absolute inset-0 bg-gradient-to-r from-primary to-secondary rounded-full blur-xl opacity-50 group-hover:opacity-75 transition-opacity duration-300 animate-pulse" />
        
        {/* Button */}
        <div className="relative bg-gradient-to-r from-primary to-secondary text-white px-5 py-3 rounded-full shadow-2xl hover:shadow-[0_0_30px_rgba(147,197,114,0.6)] transition-all duration-300 flex items-center gap-3 border border-white/20 backdrop-blur-sm">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="22"
            height="22"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className={`transition-transform duration-300 ${isHovered ? 'rotate-12 scale-110' : ''}`}
          >
            <circle cx="12" cy="12" r="10" />
            <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
            <path d="M12 17h.01" />
          </svg>
          <span className="text-sm font-semibold whitespace-nowrap">
            {isHovered ? '🚀 Começar Tour' : 'Tour Guiado'}
          </span>
        </div>
      </div>
    </button>
  );
}
