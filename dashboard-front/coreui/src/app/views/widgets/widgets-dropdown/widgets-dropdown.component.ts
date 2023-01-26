import {
  AfterContentInit,
  AfterViewInit,
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  OnInit,
  Input,
  ViewChild,
  OnChanges
} from '@angular/core';
import { getStyle } from '@coreui/utils/src';
import { ChartjsComponent } from '@coreui/angular-chartjs';

interface Metrics {
  block_rate: number;
  fraud_rate: number;
  total_revenue: number;
  chargeback_costs: number;
}

interface VariationWidget {
  block_rate: Variation;
  fraud_rate: Variation;
  total_revenue: Variation;
  chargeback_costs: Variation;
}

interface Variation {
  exist: boolean;
  percentage: number;
  arrow: string;
}

@Component({
  selector: 'app-widgets-dropdown',
  templateUrl: './widgets-dropdown.component.html',
  styleUrls: ['./widgets-dropdown.component.scss'],
  changeDetection: ChangeDetectionStrategy.Default
})
export class WidgetsDropdownComponent implements OnInit, OnChanges, AfterContentInit {

  constructor(
    private changeDetectorRef: ChangeDetectorRef
  ) {}

  @Input() originalMetrics!: Metrics;
  @Input() currentMetrics!: Metrics;
  public variations!: VariationWidget;
  public chargebackSimulation = false;

  data: any[] = [];
  options: any[] = [];
  labels = [
    'January',
    'February',
    'March',
    'April',
    'May',
    'June',
    'July',
    'August',
    'September',
    'October',
    'November',
    'December',
    'January',
    'February',
    'March',
    'April'
  ];
  datasets = [
    [{
      label: 'My First dataset',
      backgroundColor: 'transparent',
      borderColor: 'rgba(255,255,255,.55)',
      pointBackgroundColor: getStyle('--cui-danger'),
      pointHoverBorderColor: getStyle('--cui-danger'),
      data: [65, 59, 84, 84, 51, 55, 40]
    }], [{
      label: 'My Second dataset',
      backgroundColor: 'transparent',
      borderColor: 'rgba(255,255,255,.55)',
      pointBackgroundColor: getStyle('--cui-success'),
      pointHoverBorderColor: getStyle('--cui-success'),
      data: [1, 18, 9, 17, 34, 22, 11]
    }], [{
      label: 'My Third dataset',
      backgroundColor: 'rgba(255,255,255,.2)',
      borderColor: 'rgba(255,255,255,.55)',
      data: [78, 81, 80, 45, 34, 12, 40],
      barPercentage: 0.7
    }], [{
      label: 'My Fourth dataset',
      backgroundColor: 'rgba(255,255,255,.2)',
      borderColor: 'rgba(255,255,255,.55)',
      data: [78, 81, 80, 45, 34, 12, 40, 85, 65, 23, 12, 98, 34, 84, 67, 82],
      barPercentage: 0.7
    }]
  ];
  optionsDefault = {
    plugins: {
      legend: {
        display: false
      }
    },
    maintainAspectRatio: false,
    scales: {
      x: {
        grid: {
          display: false,
          drawBorder: false
        },
        ticks: {
          display: false
        }
      },
      y: {
        min: 30,
        max: 89,
        display: false,
        grid: {
          display: false
        },
        ticks: {
          display: false
        }
      }
    },
    elements: {
      line: {
        borderWidth: 1,
        tension: 0.4
      },
      point: {
        radius: 4,
        hitRadius: 10,
        hoverRadius: 4
      }
    }
  };

  ngOnInit(): void {
    this.setData();
    this.initializeVariations();
  }

  ngOnChanges(): void{
    if (this.originalMetrics.block_rate != this.currentMetrics.block_rate){
      this.variations.block_rate.exist = true;
      this.variations.block_rate.percentage = (this.currentMetrics.block_rate - this.originalMetrics.block_rate);
      if (this.variations.block_rate.percentage > 0){
        this.variations.block_rate.arrow = "cilArrowTop";
      } else {
        this.variations.block_rate.arrow = "cilArrowBottom";
      }
    } else {this.variations.block_rate.exist = false;}

    if (this.originalMetrics.fraud_rate != this.currentMetrics.fraud_rate){
      this.variations.fraud_rate.exist = true;
      this.variations.fraud_rate.percentage = (this.currentMetrics.fraud_rate - this.originalMetrics.fraud_rate);
      if (this.variations.fraud_rate.percentage > 0){
        this.variations.fraud_rate.arrow = "cilArrowTop";
      } else {
        this.variations.fraud_rate.arrow = "cilArrowBottom";
      }
    } else {this.variations.fraud_rate.exist = false;}

    if (this.originalMetrics.total_revenue != this.currentMetrics.total_revenue){
      this.variations.total_revenue.exist = true;
      this.variations.total_revenue.percentage = (this.currentMetrics.total_revenue - this.originalMetrics.total_revenue)/this.originalMetrics.total_revenue;
      if (this.variations.total_revenue.percentage > 0){
        this.variations.total_revenue.arrow = "cilArrowTop";
      } else {
        this.variations.total_revenue.arrow = "cilArrowBottom";
      }
    } else {this.variations.total_revenue.exist = false;}
    
    if (this.originalMetrics.chargeback_costs != this.currentMetrics.chargeback_costs){
      this.variations.chargeback_costs.exist = true;
      this.variations.chargeback_costs.percentage = (this.currentMetrics.chargeback_costs - this.originalMetrics.chargeback_costs)/this.originalMetrics.chargeback_costs;
      if (this.variations.chargeback_costs.percentage > 0){
        this.variations.chargeback_costs.arrow = "cilArrowTop";
        this.chargebackSimulation = true;
      }  else {
        this.variations.chargeback_costs.arrow = "cilArrowBottom";
        this.chargebackSimulation = false;
      }
    } else {this.variations.chargeback_costs.exist = false;}
  }

  initializeVariations(): void {
    this.variations = {
      block_rate : {
        exist : false,
        percentage: 0,
        arrow: ""
      },
      fraud_rate : {
        exist : false,
        percentage: 0,
        arrow: ""
      },
      total_revenue : {
        exist : false,
        percentage: 0,
        arrow: ""
      },
      chargeback_costs : {
        exist : false,
        percentage: 0,
        arrow: ""
      }
    }
  }

  ngAfterContentInit(): void {
    this.changeDetectorRef.detectChanges();

  }

  setData() {
    for (let idx = 0; idx < 4; idx++) {
      this.data[idx] = {
        labels: idx < 3 ? this.labels.slice(0, 7) : this.labels,
        datasets: this.datasets[idx]
      };
    }
    this.setOptions();
  }

  setOptions() {
    for (let idx = 0; idx < 4; idx++) {
      const options = JSON.parse(JSON.stringify(this.optionsDefault));
      switch (idx) {
        case 0: {
          this.options.push(options);
          break;
        }
        case 1: {
          options.scales.y.min = -9;
          options.scales.y.max = 39;
          this.options.push(options);
          break;
        }
        case 2: {
          options.scales.x = { display: false };
          options.scales.y = { display: false };
          options.elements.line.borderWidth = 2;
          options.elements.point.radius = 0;
          this.options.push(options);
          break;
        }
        case 3: {
          options.scales.x.grid = { display: false, drawTicks: false };
          options.scales.x.grid = { display: false, drawTicks: false, drawBorder: false };
          options.scales.y.min = undefined;
          options.scales.y.max = undefined;
          options.elements = {};
          this.options.push(options);
          break;
        }
      }
    }
  }
}

@Component({
  selector: 'app-chart-sample',
  template: '<c-chart type="line" [data]="data" [options]="options" width="300" #chart></c-chart>'
})
export class ChartSample implements AfterViewInit {

  constructor() {}

  @ViewChild('chart') chartComponent!: ChartjsComponent;

  colors = {
    label: 'My dataset',
    backgroundColor: 'rgba(77,189,116,.2)',
    borderColor: '#4dbd74',
    pointHoverBackgroundColor: '#fff'
  };

  labels = ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su'];

  data = {
    labels: this.labels,
    datasets: [{
      data: [65, 59, 84, 84, 51, 55, 40],
      ...this.colors,
      fill: { value: 65 }
    }]
  };

  options = {
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      }
    },
    elements: {
      line: {
        tension: 0.4
      }
    }
  };

  ngAfterViewInit(): void {
    setTimeout(() => {
      const data = () => {
        return {
          ...this.data,
          labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
          datasets: [{
            ...this.data.datasets[0],
            data: [42, 88, 42, 66, 77],
            fill: { value: 55 }
          }, { ...this.data.datasets[0], borderColor: '#ffbd47', data: [88, 42, 66, 77, 42] }]
        };
      };
      const newLabels = ['Jan', 'Feb', 'Mar', 'Apr', 'May'];
      const newData = [42, 88, 42, 66, 77];
      let { datasets, labels } = { ...this.data };
      // @ts-ignore
      const before = this.chartComponent?.chart?.data.datasets.length;
      console.log('before', before);
      // console.log('datasets, labels', datasets, labels)
      // @ts-ignore
      // this.data = data()
      this.data = {
        ...this.data,
        datasets: [{ ...this.data.datasets[0], data: newData }, {
          ...this.data.datasets[0],
          borderColor: '#ffbd47',
          data: [88, 42, 66, 77, 42]
        }],
        labels: newLabels
      };
      // console.log('datasets, labels', { datasets, labels } = {...this.data})
      // @ts-ignore
      setTimeout(() => {
        const after = this.chartComponent?.chart?.data.datasets.length;
        console.log('after', after);
      });
    }, 5000);
  }
}
