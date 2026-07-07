import * as React from "react";
import { TargetPanel } from "@passage/design-system";

/** Machine output sitting on the slightly lighter panel surface, once translated. */
export function WithText() {
  return (
    <TargetPanel style={{ padding: "1rem", maxWidth: "28rem" }}>
      <p style={{ margin: 0, fontFamily: "Georgia, 'Times New Roman', serif" }}>
        El informe trimestral muestra un aumento del 12% en los ingresos,
        impulsado principalmente por el crecimiento en el mercado europeo. La
        dirección espera que esta tendencia continúe en el próximo año fiscal.
      </p>
    </TargetPanel>
  );
}
