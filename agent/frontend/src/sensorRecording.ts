export interface SensorChannel {
  times: number[];
  values: number[];
  unit: string;
}

export interface UploadedSensorRecording {
  bvp: SensorChannel;
  eda: SensorChannel;
  durationSeconds: number;
}

interface ParsedChannel {
  times: number[];
  values: number[];
}

const TIME_HEADERS = new Set(["time", "timestamp", "time_seconds", "seconds"]);

function parseCsvChannel(text: string): ParsedChannel {
  const rows = text.split(/\r?\n/).map((row) => row.trim()).filter(Boolean);
  if (rows.length < 3) throw new Error("Sensor CSV has too few rows");

  const header = rows[0].split(",").map((field) => field.trim().toLowerCase());
  if (header.length >= 2 && TIME_HEADERS.has(header[0])) {
    const times: number[] = [];
    const values: number[] = [];
    for (const row of rows.slice(1)) {
      const columns = row.split(",");
      const time = Number(columns[0]);
      const value = Number(columns[1]);
      if (Number.isFinite(time) && Number.isFinite(value)) {
        times.push(time);
        values.push(value);
      }
    }
    if (times.length < 2) throw new Error("Sensor CSV has no timestamped values");
    return { times, values };
  }

  const startTime = Number(rows[0].split(",")[0]);
  const sampleRate = Number(rows[1].split(",")[0]);
  const values = rows.slice(2)
    .map((row) => Number(row.split(",")[0]))
    .filter(Number.isFinite);
  if (!Number.isFinite(startTime) || !Number.isFinite(sampleRate) || sampleRate <= 0 || !values.length) {
    throw new Error("Sensor CSV header is not recognized");
  }
  return {
    times: values.map((_, index) => startTime + index / sampleRate),
    values,
  };
}

function trimChannel(channel: ParsedChannel, start: number, end: number, unit: string): SensorChannel {
  const times: number[] = [];
  const values: number[] = [];
  channel.times.forEach((time, index) => {
    if (time >= start && time <= end) {
      times.push(time - start);
      values.push(channel.values[index]);
    }
  });
  return { times, values, unit };
}

function parseFirmwareCapture(text: string): UploadedSensorRecording {
  const times: number[] = [];
  const bvp: number[] = [];
  const eda: number[] = [];
  let firstMillis: number | undefined;
  let rolloverOffset = 0;
  let previousMillis: number | undefined;

  for (const row of text.split(/\r?\n/)) {
    const fields = row.trim().split(",");
    if (fields.length !== 4 || fields[0] !== "DATA") continue;
    const millis = Number(fields[1]);
    const bvpValue = Number(fields[2]);
    const edaValue = Number(fields[3]);
    if (![millis, bvpValue, edaValue].every(Number.isFinite)) continue;
    if (previousMillis !== undefined && previousMillis - millis > 2 ** 31) {
      rolloverOffset += 2 ** 32;
    }
    const unwrappedMillis = millis + rolloverOffset;
    firstMillis ??= unwrappedMillis;
    previousMillis = millis;
    times.push((unwrappedMillis - firstMillis) / 1000);
    bvp.push(bvpValue);
    eda.push(edaValue);
  }
  if (times.length < 2) throw new Error("Firmware capture has no sensor samples");
  return {
    bvp: { times, values: bvp, unit: "raw" },
    eda: { times, values: eda, unit: "raw" },
    durationSeconds: times[times.length - 1],
  };
}

export async function readUploadedSensorRecording(files: File[]): Promise<UploadedSensorRecording> {
  const bvpFile = files.find((file) => file.name.toLowerCase().includes("bvp"));
  const edaFile = files.find((file) => file.name.toLowerCase().includes("eda"));
  if (!bvpFile || !edaFile) {
    if (files.length !== 1) throw new Error("No sensor recording found");
    return parseFirmwareCapture(await files[0].text());
  }

  const [bvp, eda] = await Promise.all([
    bvpFile.text().then(parseCsvChannel),
    edaFile.text().then(parseCsvChannel),
  ]);
  const start = Math.max(bvp.times[0], eda.times[0]);
  const end = Math.min(bvp.times[bvp.times.length - 1], eda.times[eda.times.length - 1]);
  if (end <= start) throw new Error("BVP and EDA recordings do not overlap");
  return {
    bvp: trimChannel(bvp, start, end, "a.u."),
    eda: trimChannel(eda, start, end, "µS"),
    durationSeconds: end - start,
  };
}
