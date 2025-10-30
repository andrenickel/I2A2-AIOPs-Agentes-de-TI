import { Card } from '@/components/ui/card';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Quote } from 'lucide-react';

interface TestimonialCardProps {
  name: string;
  role: string;
  company: string;
  content: string;
  avatar?: string;
  rating?: number;
}

export function TestimonialCard({ 
  name, 
  role, 
  company, 
  content,
  avatar,
  rating = 5 
}: TestimonialCardProps) {
  const initials = name.split(' ').map(n => n[0]).join('');

  return (
    <Card className="relative bg-card/95 backdrop-blur-sm p-6 border-border/50 hover:shadow-lg transition-all duration-300 hover:-translate-y-1">
      <Quote className="absolute top-4 right-4 w-8 h-8 text-primary/20" />
      
      <div className="flex items-start gap-4 mb-4">
        <Avatar className="w-12 h-12">
          <AvatarImage src={avatar} alt={name} />
          <AvatarFallback className="bg-primary text-primary-foreground">
            {initials}
          </AvatarFallback>
        </Avatar>
        
        <div className="flex-1">
          <h4 className="font-semibold text-foreground">{name}</h4>
          <p className="text-sm text-muted-foreground">{role}</p>
          <p className="text-sm text-muted-foreground">{company}</p>
        </div>
      </div>

      <div className="flex mb-3">
        {Array.from({ length: rating }).map((_, i) => (
          <span key={i} className="text-warning">★</span>
        ))}
      </div>

      <p className="text-muted-foreground italic">"{content}"</p>
    </Card>
  );
}
