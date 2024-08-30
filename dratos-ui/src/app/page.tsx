import Image from "next/image";
import dynamic from 'next/dynamic';

const Diagram = dynamic(() => import('./components/Diagram'), { ssr: false });

export default function Home() {
  return (
    <div>
      <h1>Dratos UI</h1>
      <Diagram />
    </div>
  );
}
