import { memo } from 'react';
import { Card } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Target } from 'lucide-react';

interface Goal {
  id: number;
  title: string;
  current: number;
  target: number;
  unit: string;
}

const goals: Goal[] = [
  {
    id: 1,
    title: 'Documentos Processados',
    current: 1247,
    target: 1500,
    unit: 'docs'
  },
  {
    id: 2,
    title: 'Taxa de Automação',
    current: 97.7,
    target: 99,
    unit: '%'
  },
  {
    id: 3,
    title: 'Redução de Erros',
    current: 2.3,
    target: 1.5,
    unit: '%',
  }
];

export const GoalsCard = memo(() => {
  return (
    <Card className="bg-card/95 backdrop-blur-sm p-6 border-border/50">
      <div className="flex items-center gap-2 mb-4">
        <Target className="w-5 h-5 text-primary" />
        <h3 className="text-xl font-semibold">Metas do Mês</h3>
      </div>
      
      <div className="space-y-6">
        {goals.map((goal) => {
          const progress = (goal.current / goal.target) * 100;
          const isReversed = goal.id === 3; // Para redução de erros, menor é melhor
          const displayProgress = isReversed ? 100 - progress : progress;
          
          return (
            <div key={goal.id}>
              <div className="flex justify-between mb-2">
                <span className="text-sm font-medium">{goal.title}</span>
                <span className="text-sm text-muted-foreground">
                  {goal.current}{goal.unit} / {goal.target}{goal.unit}
                </span>
              </div>
              <Progress 
                value={displayProgress} 
                className="h-2"
              />
              <p className="text-xs text-muted-foreground mt-1">
                {displayProgress >= 100 ? '✓ Meta atingida!' : `${Math.round(displayProgress)}% concluído`}
              </p>
            </div>
          );
        })}
      </div>
    </Card>
  );
});

GoalsCard.displayName = 'GoalsCard';
